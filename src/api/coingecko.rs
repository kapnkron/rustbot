use crate::models::market::{TokenData, Candle};
use chrono::{DateTime, Utc};
use log::{error, info, warn};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::api::{MarketData as CommonMarketData, Quote};
use crate::api::types::USDData;
use crate::api::RateLimiter;
use crate::error::{Result, Error};
use std::collections::HashMap;

const API_BASE_URL: &str = "https://api.coingecko.com/api/v3";
const RATE_LIMIT: Duration = Duration::from_secs(1);
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);
const CACHE_TTL: i64 = 60; // 1 minute cache
const RATE_LIMIT_KEY: &str = "coingecko";

#[derive(Debug, Deserialize)]
struct CoinGeckoResponse<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct CoinGeckoStatus {
    error_code: Option<u32>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct CoinGeckoToken {
    id: String,
    symbol: String,
    name: String,
    market_cap_rank: Option<u32>,
    market_data: Option<CoinGeckoMarketData>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct CoinGeckoMarketData {
    current_price: Option<f64>,
    total_volume: Option<f64>,
    total_liquidity: Option<f64>,
    market_cap: Option<f64>,
    holders: Option<i64>,
    transactions_24h: Option<i64>,
}

#[derive(Debug)]
struct Cache<T> {
    data: HashMap<String, (T, Instant)>,
    ttl: Duration,
}

impl<T: Clone> Cache<T> {
    pub fn new(ttl: i64) -> Self {
        Self {
            data: HashMap::new(),
            ttl: Duration::from_secs(ttl.try_into().unwrap()),
        }
    }

    pub fn get(&self, key: &str) -> Option<T> {
        if let Some((value, timestamp)) = self.data.get(key) {
            if timestamp.elapsed() < self.ttl {
                return Some(value.clone());
            }
        }
        None
    }

    pub fn insert(&mut self, key: String, value: T) {
        self.data.insert(key, (value, Instant::now()));
    }
}

impl CoinGeckoToken {
    fn validate(&self) -> Result<()> {
        // Validate required string fields
        if self.id.trim().is_empty() {
            return Err(Error::ApiInvalidData("Empty token ID".into()));
        }
        if self.symbol.trim().is_empty() {
            return Err(Error::ApiInvalidData("Empty token symbol".into()));
        }
        if self.name.trim().is_empty() {
            return Err(Error::ApiInvalidData("Empty token name".into()));
        }

        // Validate market data if present
        if let Some(market_data) = &self.market_data {
            // Check for negative values
            if market_data.current_price.unwrap_or_default() < 0.0 {
                return Err(Error::ApiInvalidData("Negative token price".into()));
            }
            if market_data.total_volume.unwrap_or_default() < 0.0 {
                return Err(Error::ApiInvalidData("Negative trading volume".into()));
            }
            if market_data.total_liquidity.unwrap_or_default() < 0.0 {
                return Err(Error::ApiInvalidData("Negative liquidity".into()));
            }
            if market_data.market_cap.unwrap_or_default() < 0.0 {
                return Err(Error::ApiInvalidData("Negative market cap".into()));
            }
            if market_data.holders.unwrap_or_default() < 0 {
                return Err(Error::ApiInvalidData("Negative holder count".into()));
            }
            if market_data.transactions_24h.unwrap_or_default() < 0 {
                return Err(Error::ApiInvalidData("Negative transaction count".into()));
            }
        }
        Ok(())
    }

    fn into_token_data(self) -> Result<TokenData> {
        self.validate()?;
        
        let market_data = self.market_data
            .ok_or_else(|| Error::ApiInvalidData("Missing market data".to_string()))?;
            
        Ok(TokenData {
            address: self.id,
            symbol: self.symbol.to_uppercase(),
            name: self.name,
            price: market_data.current_price.unwrap_or_default(),
            volume_24h: market_data.total_volume.unwrap_or_default(),
            liquidity: market_data.total_liquidity.unwrap_or_default(),
            market_cap: market_data.market_cap.unwrap_or_default(),
            holders: market_data.holders.unwrap_or_default() as u64,
            transactions_24h: market_data.transactions_24h.unwrap_or_default() as u64,
            last_updated: Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceHistory {
    pub timestamps: Vec<i64>,
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
}

impl PriceHistory {
    pub fn validate(&self) -> Result<()> {
        // Check for empty data
        if self.timestamps.is_empty() || self.prices.is_empty() || self.volumes.is_empty() {
            return Err(Error::ApiInvalidData("Empty price history data".into()));
        }

        // Check all arrays have same length
        if self.timestamps.len() != self.prices.len() || self.timestamps.len() != self.volumes.len() {
            return Err(Error::ApiInvalidData("Mismatched array lengths in price history".into()));
        }

        // Validate timestamps are in ascending order
        for i in 1..self.timestamps.len() {
            if self.timestamps[i] <= self.timestamps[i-1] {
                return Err(Error::ApiInvalidData("Timestamps not in ascending order".into()));
            }
        }

        // Validate prices and volumes are non-negative
        if self.prices.iter().any(|&p| p < 0.0) || self.volumes.iter().any(|&v| v < 0.0) {
            return Err(Error::ApiInvalidData("Negative prices or volumes found".into()));
        }

        Ok(())
    }

    pub fn into_candles(&self) -> Result<Vec<Candle>> {
        self.validate()?;
        
        let mut candles = Vec::with_capacity(self.timestamps.len());
        
        for i in 0..self.timestamps.len() {
            // Additional validation for each data point
            if !self.prices[i].is_finite() {
                return Err(Error::ApiInvalidData(format!(
                    "Invalid price value at index {}: {}", i, self.prices[i]
                )));
            }
            if !self.volumes[i].is_finite() {
                return Err(Error::ApiInvalidData(format!(
                    "Invalid volume value at index {}: {}", i, self.volumes[i]
                )));
            }
            
            candles.push(Candle {
                timestamp: DateTime::from_timestamp(self.timestamps[i], 0).unwrap_or(Utc::now()),
                open: self.prices[i],
                high: self.prices[i],
                low: self.prices[i],
                close: self.prices[i],
                volume: self.volumes[i],
                pair: "".to_string(), // You'll need to provide the pair somehow
            });
        }
        
        Ok(candles)
    }
}

impl From<CoinGeckoToken> for CommonMarketData {
    fn from(token: CoinGeckoToken) -> CommonMarketData {
        let market_data = token.market_data.unwrap_or_default();
        CommonMarketData {
            symbol: token.symbol.to_uppercase(),
            price: market_data.current_price.unwrap_or_default(),
            volume: market_data.total_volume.unwrap_or_default(),
            market_cap: market_data.market_cap.unwrap_or_default(),
            price_change_24h: 0.0, // Not available in CoinGeckoToken
            volume_change_24h: 0.0, // Not available in CoinGeckoToken
            timestamp: Utc::now(),
            volume_24h: market_data.total_volume.unwrap_or_default(),
            change_24h: 0.0, // Not available in CoinGeckoToken
            quote: Quote {
                usd: USDData {
                    price: market_data.current_price.unwrap_or_default(),
                    volume_24h: market_data.total_volume.unwrap_or_default(),
                    market_cap: market_data.market_cap.unwrap_or_default(),
                    percent_change_24h: 0.0, // Not available in CoinGeckoToken
                    volume_change_24h: 0.0, // Not available in CoinGeckoToken
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct CoinGeckoClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
    last_request: Instant,
    token_cache: Cache<CoinGeckoToken>,
    price_cache: Cache<PriceHistory>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl CoinGeckoClient {
    pub fn new(api_key: Option<String>, rate_limiter: Arc<Mutex<RateLimiter>>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.coingecko.com/api/v3".to_string(),
            last_request: Instant::now(),
            token_cache: Cache::new(CACHE_TTL),
            price_cache: Cache::new(CACHE_TTL),
            rate_limiter,
        }
    }

    async fn wait_for_rate_limit(&mut self) -> Result<()> {
        let elapsed = self.last_request.elapsed();
        if elapsed < RATE_LIMIT {
            let wait_time = RATE_LIMIT - elapsed;
            info!("Rate limit wait: {}ms", wait_time.as_millis());
            tokio::time::sleep(wait_time).await;
        }
        self.last_request = Instant::now();
        Ok(())
    }

    async fn check_rate_limit(&self) -> Result<()> {
        let limiter = self.rate_limiter.lock().await;
        let can_proceed = limiter.check(RATE_LIMIT_KEY, RATE_LIMIT).await;
        if !can_proceed {
            return Err(Error::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
        }
        Ok(())
    }

    async fn make_request<T: for<'de> Deserialize<'de>>(
        &mut self,
        endpoint: &str,
        params: &[(&str, &str)],
        retry_count: u32,
    ) -> Result<T> {
        // Check rate limit with exponential backoff
        let backoff_duration = if retry_count > 0 {
            Duration::from_secs(2u64.pow(retry_count))
        } else {
            RATE_LIMIT
        };

        // Check rate limit before making request
        let can_proceed = {
            let limiter = self.rate_limiter.lock().await;
            limiter.check(RATE_LIMIT_KEY, RATE_LIMIT).await
        };
        
        if !can_proceed {
            if retry_count < MAX_RETRIES {
                warn!("Rate limit exceeded, backing off for {} seconds...", backoff_duration.as_secs());
                tokio::time::sleep(backoff_duration).await;
                return Box::pin(self.make_request::<T>(endpoint, params, retry_count + 1)).await;
            }
            return Err(Error::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
        }

        // Build request URL with parameters
        let url = format!(
            "{}/{}{}",
            self.base_url,
            endpoint,
            if params.is_empty() {
                String::new()
            } else {
                format!(
                    "?{}",
                    params.iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join("&")
                )
            }
        );

        // Make the request with proper error handling
        let mut request = self.client.get(&url);
        if let Some(key) = &self.api_key {
            request = request.header("X-CG-Pro-API-Key", key);
        }

        let _response = request.send().await?;

        // Handle response based on status code
        match _response.status() {
            status if status.is_success() => {
                let cg_response: CoinGeckoResponse<T> = match _response.json().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        error!("Failed to parse response: {}", e);
                        return Err(Error::ApiInvalidFormat(format!("Failed to parse response: {}", e)));
                    }
                };

                Ok(cg_response.data)
            }
            status => {
                if status == StatusCode::TOO_MANY_REQUESTS {
                    return Err(Error::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
                }
                let error = match status.as_u16() {
                    429 => {
                        if retry_count < MAX_RETRIES {
                            warn!("Rate limit exceeded, backing off for {} seconds...", backoff_duration.as_secs());
                            tokio::time::sleep(backoff_duration).await;
                            return Box::pin(self.make_request::<T>(endpoint, params, retry_count + 1)).await;
                        }
                        Error::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string())
                    }
                    401 => Error::ApiAuthFailed("Authentication failed".into()),
                    403 => Error::ApiQuotaExceeded("API quota exceeded".into()),
                    503 => Error::ApiMaintenance("API is under maintenance".into()),
                    _ => {
                        let error_msg = format!("API request failed with status: {}", status);
                        error!("{}", error_msg);
                        Error::ApiError(error_msg)
                    }
                };
                Err(error)
            }
        }
    }

    pub async fn get_token_info(&mut self, token_id: &str) -> Result<TokenData> {
        if let Some(cached) = self.token_cache.get(token_id) {
            info!("Using cached token info for: {}", token_id);
            return cached.into_token_data();
        }

        let token_info = self.make_request::<CoinGeckoToken>(
            &format!("coins/{}", token_id),
            &[("localization", "false"), ("tickers", "false"), ("market_data", "true")],
            0
        ).await?;
        
        token_info.validate()?;
        self.token_cache.insert(token_id.to_string(), token_info.clone());
        
        token_info.into_token_data()
    }

    pub async fn get_price_history(
        &mut self,
        token_id: &str,
        days: u32,
    ) -> Result<PriceHistory> {
        let history = self.make_request::<PriceHistory>(
            &format!("coins/{}/market_chart", token_id),
            &[("vs_currency", "usd"), ("days", &days.to_string())],
            0
        ).await?;

        history.validate()?;
        Ok(history)
    }

    pub async fn search_tokens(&mut self, query: &str) -> Result<Vec<CoinGeckoToken>> {
        let tokens = self.make_request::<Vec<CoinGeckoToken>>(
            "search",
            &[("query", query)],
            0
        ).await?;
        
        // Validate each token
        for token in &tokens {
            token.validate()?;
        }
        
        Ok(tokens)
    }

    pub async fn get_trending_tokens(&mut self) -> Result<Vec<CoinGeckoToken>> {
        let tokens = self.make_request::<Vec<CoinGeckoToken>>(
            "search/trending",
            &[],
            0
        ).await?;
        
        // Validate each token
        for token in &tokens {
            token.validate()?;
        }
        
        Ok(tokens)
    }

    pub async fn get_quote_from_symbol(&self, symbol: &str) -> Result<Quote> {
        // Implementation of get_quote_from_symbol method
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_quotes_from_symbols(&self, symbols: &[&str]) -> Result<Vec<String>> {
        // Implementation of get_quotes_from_symbols method
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_exchanges(&self) -> Result<Vec<String>> {
        // Implementation of get_exchanges method
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<CommonMarketData> {
        let response = self.client.get(&format!("{}/coins/{}", self.base_url, symbol))
            .header("X-CG-Pro-API-Key", self.api_key.as_ref().unwrap_or(&"".to_string()))
            .send()
            .await?;

        let data: CoinGeckoResponse<CommonMarketData> = response.json().await?;
        Ok(data.data)
    }
} 
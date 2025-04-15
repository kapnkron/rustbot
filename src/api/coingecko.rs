use crate::models::market::{TokenData, Candle};
use chrono::{DateTime, Utc};
use log::{error, info, warn};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::api::{TokenInfo, MarketData, ApiError, Quote};
use crate::web::validation::RateLimiter;
use crate::utils::error::Result;

const API_BASE_URL: &str = "https://api.coingecko.com/api/v3";
const RATE_LIMIT: Duration = Duration::from_secs(1);
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);
const CACHE_TTL: i64 = 60; // 1 minute cache
const RATE_LIMIT_KEY: &str = "coingecko";

#[derive(Debug, Deserialize)]
struct CoinGeckoResponse<T> {
    data: Option<T>,
    error: Option<String>,
    status: Option<CoinGeckoStatus>,
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

#[derive(Debug, Deserialize, Clone)]
struct CoinGeckoMarketData {
    current_price: Option<f64>,
    total_volume: Option<f64>,
    total_liquidity: Option<f64>,
    market_cap: Option<f64>,
    holders: Option<i64>,
    transactions_24h: Option<i64>,
}

// Define a local Cache type since we can't find the imported one
struct Cache<T> {
    ttl: i64,
    cache: std::collections::HashMap<String, (T, Instant)>,
}

impl<T: Clone> Cache<T> {
    fn new(ttl: i64) -> Self {
        Self {
            ttl,
            cache: std::collections::HashMap::new(),
        }
    }

    async fn get(&self, key: &str) -> Option<T> {
        if let Some((value, timestamp)) = self.cache.get(key) {
            if timestamp.elapsed().as_secs() < self.ttl as u64 {
                return Some(value.clone());
            }
        }
        None
    }

    async fn set(&mut self, key: String, value: T) {
        self.cache.insert(key, (value, Instant::now()));
    }
}

// Local Result and Error types since we can't find the imported ones
type Result<T> = std::result::Result<T, TradingError>;

#[derive(Debug)]
enum TradingError {
    ApiError(String),
    ApiInvalidData(String),
    ApiInvalidFormat(String),
    ApiConnectionFailed(String),
    ApiAuthFailed(String),
    ApiQuotaExceeded(String),
    ApiMaintenance(String),
    RateLimitExceeded(String),
}

impl CoinGeckoToken {
    fn validate(&self) -> Result<()> {
        // Validate required string fields
        if self.id.trim().is_empty() {
            return Err(TradingError::ApiInvalidData("Empty token ID".into()));
        }
        if self.symbol.trim().is_empty() {
            return Err(TradingError::ApiInvalidData("Empty token symbol".into()));
        }
        if self.name.trim().is_empty() {
            return Err(TradingError::ApiInvalidData("Empty token name".into()));
        }

        // Validate market data if present
        if let Some(market_data) = &self.market_data {
            // Check for negative values
            if market_data.current_price.unwrap_or_default() < 0.0 {
                return Err(TradingError::ApiInvalidData("Negative token price".into()));
            }
            if market_data.total_volume.unwrap_or_default() < 0.0 {
                return Err(TradingError::ApiInvalidData("Negative trading volume".into()));
            }
            if market_data.total_liquidity.unwrap_or_default() < 0.0 {
                return Err(TradingError::ApiInvalidData("Negative liquidity".into()));
            }
            if market_data.market_cap.unwrap_or_default() < 0.0 {
                return Err(TradingError::ApiInvalidData("Negative market cap".into()));
            }
            if market_data.holders.unwrap_or_default() < 0 {
                return Err(TradingError::ApiInvalidData("Negative holder count".into()));
            }
            if market_data.transactions_24h.unwrap_or_default() < 0 {
                return Err(TradingError::ApiInvalidData("Negative transaction count".into()));
            }
        }
        Ok(())
    }

    fn into_token_data(self) -> Result<TokenData> {
        self.validate()?;
        
        let market_data = self.market_data
            .ok_or_else(|| TradingError::ApiInvalidData("Missing market data".to_string()))?;
            
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
            return Err(TradingError::ApiInvalidData("Empty price history data".into()));
        }

        // Check all arrays have same length
        if self.timestamps.len() != self.prices.len() || self.timestamps.len() != self.volumes.len() {
            return Err(TradingError::ApiInvalidData("Mismatched array lengths in price history".into()));
        }

        // Validate timestamps are in ascending order
        for i in 1..self.timestamps.len() {
            if self.timestamps[i] <= self.timestamps[i-1] {
                return Err(TradingError::ApiInvalidData("Timestamps not in ascending order".into()));
            }
        }

        // Validate prices and volumes are non-negative
        if self.prices.iter().any(|&p| p < 0.0) || self.volumes.iter().any(|&v| v < 0.0) {
            return Err(TradingError::ApiInvalidData("Negative prices or volumes found".into()));
        }

        Ok(())
    }

    pub fn into_candles(&self) -> Result<Vec<Candle>> {
        self.validate()?;
        
        let mut candles = Vec::with_capacity(self.timestamps.len());
        
        for i in 0..self.timestamps.len() {
            // Additional validation for each data point
            if !self.prices[i].is_finite() {
                return Err(TradingError::ApiInvalidData(format!(
                    "Invalid price value at index {}: {}", i, self.prices[i]
                )));
            }
            if !self.volumes[i].is_finite() {
                return Err(TradingError::ApiInvalidData(format!(
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
}

pub struct CoinGeckoClient {
    client: Client,
    api_key: String,
    base_url: String,
    last_request: Instant,
    token_cache: Cache<CoinGeckoToken>,
    price_cache: Cache<PriceHistory>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl CoinGeckoClient {
    pub fn new(api_key: String, rate_limiter: Arc<Mutex<RateLimiter>>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
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
        if !limiter.check(RATE_LIMIT_KEY, RATE_LIMIT) {
            return Err(TradingError::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
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
        let limiter = self.rate_limiter.lock().await;
        if !limiter.check(RATE_LIMIT_KEY, RATE_LIMIT) {
            if retry_count < MAX_RETRIES {
                warn!("Rate limit exceeded, backing off for {} seconds...", backoff_duration.as_secs());
                tokio::time::sleep(backoff_duration).await;
                return self.make_request::<T>(endpoint, params, retry_count + 1).await;
            }
            return Err(TradingError::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
        }
        drop(limiter);

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
        let response = match self.client
            .get(&url)
            .header("x-cg-pro-api-key", &self.api_key)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!("Request failed: {}", e);
                return Err(TradingError::ApiConnectionFailed(format!("Request failed: {}", e)));
            }
        };

        // Handle response based on status code
        match response.status() {
            status if status.is_success() => {
                let cg_response: CoinGeckoResponse<T> = match response.json().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        error!("Failed to parse response: {}", e);
                        return Err(TradingError::ApiInvalidFormat(format!("Failed to parse response: {}", e)));
                    }
                };

                // Handle API-level errors
                if let Some(error) = cg_response.error {
                    error!("API error: {}", error);
                    return Err(TradingError::ApiError(format!("API error: {}", error)));
                }

                // Handle status-level errors
                if let Some(status) = cg_response.status {
                    if let Some(error_code) = status.error_code {
                        let error = match error_code {
                            429 => {
                                if retry_count < MAX_RETRIES {
                                    warn!("Rate limit exceeded, backing off for {} seconds...", backoff_duration.as_secs());
                                    tokio::time::sleep(backoff_duration).await;
                                    return self.make_request::<T>(endpoint, params, retry_count + 1).await;
                                }
                                TradingError::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string())
                            }
                            401 => TradingError::ApiAuthFailed("Authentication failed".into()),
                            403 => TradingError::ApiQuotaExceeded("API quota exceeded".into()),
                            503 => TradingError::ApiMaintenance("API is under maintenance".into()),
                            _ => {
                                if let Some(error_msg) = status.error_message {
                                    TradingError::ApiError(format!("API error: {}", error_msg))
                                } else {
                                    TradingError::ApiError(format!("Unknown error code: {}", error_code))
                                }
                            }
                        };
                        return Err(error);
                    }
                }

                cg_response.data.ok_or_else(|| TradingError::ApiInvalidData("Empty response data".into()))
            }
            status => {
                if status == StatusCode::TOO_MANY_REQUESTS {
                    return Err(TradingError::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
                }
                let error = match status.as_u16() {
                    429 => {
                        if retry_count < MAX_RETRIES {
                            warn!("Rate limit exceeded, backing off for {} seconds...", backoff_duration.as_secs());
                            tokio::time::sleep(backoff_duration).await;
                            return self.make_request::<T>(endpoint, params, retry_count + 1).await;
                        }
                        TradingError::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string())
                    }
                    401 => TradingError::ApiAuthFailed("Authentication failed".into()),
                    403 => TradingError::ApiQuotaExceeded("API quota exceeded".into()),
                    503 => TradingError::ApiMaintenance("API is under maintenance".into()),
                    _ => {
                        let error_msg = format!("API request failed with status: {}", status);
                        error!("{}", error_msg);
                        TradingError::ApiError(error_msg)
                    }
                };
                Err(error)
            }
        }
    }

    pub async fn get_token_info(&mut self, token_id: &str) -> Result<TokenData> {
        if let Some(cached) = self.token_cache.get(token_id).await {
            info!("Using cached token info for: {}", token_id);
            return cached.into_token_data();
        }

        let token_info = self.make_request::<CoinGeckoToken>(
            &format!("coins/{}", token_id),
            &[("localization", "false"), ("tickers", "false"), ("market_data", "true")],
            0
        ).await?;
        
        token_info.validate()?;
        self.token_cache.set(token_id.to_string(), token_info.clone()).await;
        
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
        Err(TradingError::ApiError("Not implemented".to_string()))
    }

    pub async fn get_quotes_from_symbols(&self, symbols: &[&str]) -> Result<Vec<String>> {
        // Implementation of get_quotes_from_symbols method
        Err(TradingError::ApiError("Not implemented".to_string()))
    }

    pub async fn get_exchanges(&self) -> Result<Vec<String>> {
        // Implementation of get_exchanges method
        Err(TradingError::ApiError("Not implemented".to_string()))
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<MarketData> {
        let url = format!("{}/simple/price", self.base_url);
        let params = [
            ("ids", symbol),
            ("vs_currencies", "usd"),
            ("include_24hr_vol", "true"),
            ("include_24hr_change", "true"),
            ("include_market_cap", "true"),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .header("x-cg-pro-api-key", &self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(crate::utils::error::TradingError::ApiError(format!(
                "CoinGecko API error: {}",
                response.status()
            )).into());
        }

        let data: serde_json::Value = response.json().await?;
        
        // Extract data from response
        let price = data[symbol]["usd"].as_f64().unwrap_or(0.0);
        let volume = data[symbol]["usd_24h_vol"].as_f64().unwrap_or(0.0);
        let market_cap = data[symbol]["usd_market_cap"].as_f64().unwrap_or(0.0);
        let price_change_24h = data[symbol]["usd_24h_change"].as_f64().unwrap_or(0.0);
        
        // Calculate volume change (not directly provided by API)
        let volume_change_24h = 0.0; // TODO: Implement volume change calculation

        Ok(MarketData {
            price,
            volume,
            market_cap,
            price_change_24h,
            volume_change_24h,
        })
    }
} 
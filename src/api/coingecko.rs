use crate::models::market::Candle;
use chrono::{DateTime, Utc};
use log::{error, info};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::api::types::{MarketData as CommonMarketData, Quote, PriceHistory};
use crate::api::types::USDData;
use crate::security::rate_limit::RateLimiter;
use crate::error::{Result, Error};
use std::collections::HashMap;
use crate::utils::cache::Cache;
use crate::api::ApiError;

const API_BASE_URL: &str = "https://api.coingecko.com/api/v3";
// const RATE_LIMIT: Duration = Duration::from_secs(1); // Commented out
// const MAX_RETRIES: u32 = 3; // Commented out
const CACHE_TTL: i64 = 60; 
const RATE_LIMIT_KEY: &str = "coingecko";

#[derive(Debug, Deserialize)]
struct CoinGeckoResponse<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct CoinGeckoStatus { 
    _error_code: Option<u32>,
    _error_message: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CoinGeckoToken {
    pub(crate) _id: String, // Mark as unused
    pub(crate) symbol: String, // Keep symbol as it's used in From impl
    pub(crate) _name: String, // Mark as unused
    pub(crate) _market_cap_rank: Option<u32>,
    pub(crate) market_data: Option<CoinGeckoMarketData>, // Keep as used in From impl
}

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub(crate) struct CoinGeckoMarketData {
    pub(crate) current_price: Option<f64>, // Keep
    pub(crate) total_volume: Option<f64>, // Keep
    pub(crate) _total_liquidity: Option<f64>,
    pub(crate) market_cap: Option<f64>, // Keep
    pub(crate) _holders: Option<i64>,
    pub(crate) _transactions_24h: Option<i64>,
}

// Comment out unused impl block if methods inside are unused
// impl CoinGeckoToken {
//     fn validate(&self) -> Result<()> { ... }
//     fn into_token_data(self) -> Result<TokenData> { ... }
// }

impl PriceHistory {
    pub fn validate(&self) -> Result<()> {
        // Use correct field names: prices, total_volumes
        if self.prices.is_empty() || self.total_volumes.is_empty() {
            return Err(Error::ApiInvalidData("Empty price history data".into()));
        }

        // Check prices and volumes have same length (assuming timestamp comes from prices)
        if self.prices.len() != self.total_volumes.len() {
            return Err(Error::ApiInvalidData("Mismatched array lengths in price history".into()));
        }

        // Validate timestamps are in ascending order (using index 0 of prices)
        for i in 1..self.prices.len() {
            if self.prices[i][0] <= self.prices[i-1][0] { // Check timestamp part
                return Err(Error::ApiInvalidData("Timestamps not in ascending order".into()));
            }
        }

        // Validate prices (index 1) and volumes (index 1) are non-negative
        if self.prices.iter().any(|p| p[1] < 0.0) || self.total_volumes.iter().any(|v| v[1] < 0.0) {
            return Err(Error::ApiInvalidData("Negative prices or volumes found".into()));
        }

        Ok(())
    }

    pub fn into_candles(&self) -> Result<Vec<Candle>> {
        self.validate()?;
        
        let mut candles = Vec::with_capacity(self.prices.len()); // Use prices length
        
        for i in 0..self.prices.len() { // Use prices length
            let timestamp_ms = self.prices[i][0] as i64; // Timestamp from prices[0]
            let price = self.prices[i][1]; // Price from prices[1]
            let volume = self.total_volumes[i][1]; // Volume from total_volumes[1]

            // Additional validation for each data point
            if !price.is_finite() {
                return Err(Error::ApiInvalidData(format!(
                    "Invalid price value at index {}: {}", i, price
                )));
            }
            if !volume.is_finite() {
                return Err(Error::ApiInvalidData(format!(
                    "Invalid volume value at index {}: {}", i, volume
                )));
            }
            
            candles.push(Candle {
                // Convert ms timestamp to DateTime<Utc>
                timestamp: DateTime::from_timestamp_millis(timestamp_ms).ok_or_else(|| Error::ApiInvalidData("Invalid timestamp".to_string()))?,
                open: price, // Use price for O, H, L, C as we only have one value per timestamp
                high: price,
                low: price,
                close: price,
                volume, 
                pair: "".to_string(), 
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
    base_url: String,
    client: Client,
    api_key: Option<String>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    _last_request: Instant,
    token_cache: Cache<CoinGeckoToken>,
    price_cache: Cache<PriceHistory>,
}

impl CoinGeckoClient {
     pub fn new(api_key: Option<String>, rate_limiter: Arc<Mutex<RateLimiter>>) -> Self {
        Self {
            base_url: API_BASE_URL.to_string(),
            client: Client::new(),
            api_key,
            rate_limiter,
            _last_request: Instant::now(), 
            token_cache: Cache::new(CACHE_TTL),
            price_cache: Cache::new(CACHE_TTL),
        }
    }

    async fn check_rate_limit_coingecko(&self) -> Result<()> {
        let limiter = self.rate_limiter.lock().await;
        if !limiter.check(RATE_LIMIT_KEY).await? {
            return Err(Error::RateLimitExceeded("CoinGecko API rate limit exceeded".to_string()));
        }
        Ok(())
    }

    async fn make_request<T: for<'de> Deserialize<'de>>(
        &mut self,
        endpoint: &str,
        params_opt: Option<HashMap<String, String>>,
    ) -> Result<T> {
        self.check_rate_limit_coingecko().await?;

        let mut url = format!("{}{}", self.base_url, endpoint);
        if let Some(params) = params_opt {
            if !params.is_empty() {
                 url.push('?');
                 let query_string = serde_urlencoded::to_string(&params).map_err(|e| Error::InternalError(e.to_string()))?;
                 url.push_str(&query_string);
            }
        }

        let mut request_builder = self.client.get(&url);
        if let Some(key) = &self.api_key {
             request_builder = request_builder.header("x-cg-demo-api-key", key);
        }

        match request_builder.send().await {
            Ok(response) => {
                 match response.status() {
                     StatusCode::OK => response.json::<T>().await.map_err(|e| e.into()),
                     StatusCode::TOO_MANY_REQUESTS => Err(Error::RateLimitExceeded("CoinGecko API rate limit exceeded (429)".to_string())),
                     status => {
                         let body = response.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                         error!("CoinGecko API Error: Status {}, Body: {}", status, body);
                         Err(ApiError::RequestError(format!("Request failed with status: {}", status)).into())
                     }
                 }
            }
             Err(e) => {
                 error!("Reqwest Error for CoinGecko: {}", e);
                 Err(e.into())
             }
        }
    }

    pub async fn get_token_info(&mut self, token_id: &str) -> Result<CoinGeckoToken> {
        let cache_key = format!("token_info_{}", token_id);
        if let Some(cached) = self.token_cache.get(&cache_key).await {
            info!("Using cached token info for: {}", token_id);
            return Ok(cached);
        }

        let params = HashMap::from([
            ("localization".to_string(), "false".to_string()),
            ("tickers".to_string(), "false".to_string()),
            ("market_data".to_string(), "true".to_string()),
        ]);

        let token_info = self.make_request::<CoinGeckoToken>(
            &format!("coins/{}", token_id),
            Some(params),
        ).await?;
        
        self.token_cache.set(cache_key, token_info.clone()).await;
        
        Ok(token_info)
    }

    pub async fn get_price_history(&mut self, token_id: &str, days: u32) -> Result<PriceHistory> {
        let cache_key = format!("price_history_{}_{}", token_id, days);
        if let Some(cached) = self.price_cache.get(&cache_key).await {
            info!("Using cached price history for: {} ({} days)", token_id, days);
            return Ok(cached);
        }

        let params = HashMap::from([
            ("vs_currency".to_string(), "usd".to_string()),
            ("days".to_string(), days.to_string()), 
        ]);

        let history = self.make_request::<PriceHistory>(
            &format!("coins/{}/market_chart", token_id),
            Some(params),
        ).await?;

        self.price_cache.set(cache_key, history.clone()).await;
        Ok(history)
    }

    pub async fn search_tokens(&mut self, query: &str) -> Result<Vec<CoinGeckoToken>> {
        let params = HashMap::from([("query".to_string(), query.to_string())]);
        let response: serde_json::Value = self.make_request(
            "search",
            Some(params),
        ).await?;
        let tokens = serde_json::from_value(response["coins"].clone())?;
        Ok(tokens)
    }

    pub async fn get_trending_tokens(&mut self) -> Result<Vec<CoinGeckoToken>> {
        let response: serde_json::Value = self.make_request("search/trending", None).await?;
        let items = response["coins"].as_array()
            .ok_or_else(|| Error::ApiInvalidFormat("Missing 'coins' array in trending response".into()))?;
        
        let mut tokens = Vec::new();
        for item_value in items {
            let item = item_value["item"].clone();
            let token: CoinGeckoToken = serde_json::from_value(item).map_err(|e| Error::ParseError(e.to_string()))?;
            tokens.push(token);
        }
        Ok(tokens)
    }

    pub async fn get_quote_from_symbol(&self, _symbol: &str) -> Result<Quote> {
        // Implementation of get_quote_from_symbol method
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_quotes_from_symbols(&self, _symbols: &[&str]) -> Result<Vec<String>> {
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_exchanges(&self) -> Result<Vec<String>> {
        Err(Error::ApiError("Not implemented".to_string()))
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<CommonMarketData> {
        let response = self.client.get(format!("{}/coins/{}", self.base_url, symbol))
            .send()
            .await?;

        let response_data: CoinGeckoResponse<CommonMarketData> = response.json().await?;
        Ok(response_data.data)
    }
} 
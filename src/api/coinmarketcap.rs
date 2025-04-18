use crate::api::{MarketData as ApiMarketData, Quote, ApiError, RateLimiter};
use crate::utils::cache::Cache;
use crate::utils::error::{Result, Error};
use chrono::{DateTime, Utc};
use log::{info};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;

const API_BASE_URL: &str = "https://pro-api.coinmarketcap.com/v1";
const RATE_LIMIT: Duration = Duration::from_secs(1);
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);
const CACHE_TTL: i64 = 60; // 1 minute cache
const RATE_LIMIT_KEY: &str = "coinmarketcap";

#[derive(Debug, Deserialize)]
struct CMCResponse<T> {
    status: CMCStatus,
    data: Option<T>,
}

#[derive(Debug, Deserialize)]
struct CMCStatus {
    error_code: u32,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct CMCToken {
    id: u32,
    name: String,
    symbol: String,
    slug: String,
    quote: Option<CMCTokenQuote>,
}

#[derive(Debug, Deserialize, Clone)]
struct CMCTokenQuote {
    #[serde(rename = "USD")]
    usd: Option<CMCTokenQuoteUSD>,
}

#[derive(Debug, Deserialize, Clone)]
struct CMCTokenQuoteUSD {
    price: f64,
    volume_24h: f64,
    market_cap: f64,
    percent_change_1h: f64,
    percent_change_24h: f64,
    percent_change_7d: f64,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Clone)]
struct CMCTokenList {
    data: Vec<CMCToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CoinMarketCapResponse {
    data: Vec<CoinMarketCapData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CoinMarketCapData {
    quote: Quote,
}

#[derive(Debug, Serialize, Deserialize)]
struct USDData {
    price: f64,
    volume_24h: f64,
    market_cap: f64,
    percent_change_24h: f64,
    volume_change_24h: f64,
}

#[derive(Debug, Clone)]
pub struct CoinMarketCapClient {
    client: Client,
    api_key: String,
    base_url: String,
    last_request: Instant,
    token_cache: Cache<CMCToken>,
    quote_cache: Cache<CMCTokenQuote>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl CoinMarketCapClient {
    pub fn new(api_key: String, rate_limiter: Arc<Mutex<RateLimiter>>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
            api_key,
            base_url: "https://pro-api.coinmarketcap.com/v1".to_string(),
            last_request: Instant::now(),
            token_cache: Cache::new(CACHE_TTL),
            quote_cache: Cache::new(CACHE_TTL),
            rate_limiter,
        }
    }

    async fn wait_for_rate_limit(&mut self) {
        let elapsed = self.last_request.elapsed();
        if elapsed < RATE_LIMIT {
            tokio::time::sleep(RATE_LIMIT - elapsed).await;
        }
        self.last_request = Instant::now();
    }

    async fn make_request<T: for<'de> Deserialize<'de>>(
        &mut self,
        endpoint: &str,
        params: &[(&str, &str)],
        retry_count: u32,
    ) -> Result<T> {
        let url = format!("{}{}", API_BASE_URL, endpoint);
        let mut request = self.client
            .get(&url)
            .header("X-CMC_PRO_API_KEY", &self.api_key);

        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }

        match request.send().await {
            Ok(response) => {
                match response.status() {
                    StatusCode::OK => {
                        response.json::<T>().await.map_err(|e| e.into())
                    }
                    StatusCode::TOO_MANY_REQUESTS if retry_count < MAX_RETRIES => {
                        tokio::time::sleep(Duration::from_secs(2u64.pow(retry_count))).await;
                        Box::pin(self.make_request(endpoint, params, retry_count + 1)).await
                    }
                    status => {
                        Err(ApiError::RequestError(format!(
                            "Request failed with status: {}", status
                        )).into())
                    }
                }
            }
            Err(e) => {
                if retry_count < MAX_RETRIES {
                    tokio::time::sleep(Duration::from_secs(2u64.pow(retry_count))).await;
                    Box::pin(self.make_request(endpoint, params, retry_count + 1)).await
                } else {
                    Err(e.into())
                }
            }
        }
    }

    pub async fn get_token_info(&mut self, token_id: &str) -> Result<CMCToken> {
        if let Some(cached) = self.token_cache.get(token_id).await {
            info!("Using cached token info for: {}", token_id);
            return Ok(cached);
        }

        let token_info = self.make_request::<CMCToken>(
            "cryptocurrency/info",
            &[("id", token_id)],
            0
        ).await?;
        
        self.token_cache.set(token_id.to_string(), token_info.clone()).await;
        
        Ok(token_info)
    }

    pub async fn get_token_quote(&mut self, token_id: &str) -> Result<CMCTokenQuote> {
        if let Some(cached) = self.quote_cache.get(token_id).await {
            info!("Using cached quote for: {}", token_id);
            return Ok(cached);
        }

        let quote = self.make_request::<CMCTokenQuote>(
            "cryptocurrency/quotes/latest",
            &[("id", token_id), ("convert", "USD")],
            0
        ).await?;
        
        self.quote_cache.set(token_id.to_string(), quote.clone()).await;
        
        Ok(quote)
    }

    pub async fn get_top_tokens(&mut self, limit: u32) -> Result<Vec<CMCToken>> {
        let response = self.make_request::<CMCTokenList>(
            "cryptocurrency/listings/latest",
            &[
                ("start", "1"),
                ("limit", &limit.to_string()),
                ("convert", "USD"),
            ],
            0
        ).await?;
        
        Ok(response.data)
    }

    pub async fn get_trending_tokens(&mut self) -> Result<Vec<CMCToken>> {
        let response = self.make_request::<CMCTokenList>(
            "cryptocurrency/trending/latest",
            &[],
            0
        ).await?;
        
        Ok(response.data)
    }

    pub async fn get_token_quote_from_symbol(&self, symbol: &str) -> Result<Quote> {
        // Check rate limit
        let limiter = self.rate_limiter.lock().await;
        if !limiter.check(RATE_LIMIT_KEY, RATE_LIMIT).await {
            return Err(Error::RateLimitExceeded("API rate limit exceeded".to_string()));
        }

        let url = format!("{}/cryptocurrency/quotes/latest", API_BASE_URL);
        let response = self.client
            .get(&url)
            .header("X-CMC_PRO_API_KEY", &self.api_key)
            .query(&[("symbol", symbol), ("convert", "USD")])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError(format!(
                "CoinMarketCap API error: {}",
                response.status()
            )).into());
        }

        let data: serde_json::Value = response.json().await?;
        
        let quote = data["data"][symbol]["quote"].clone();
        Ok(serde_json::from_value(quote)?)
    }

    pub async fn get_top_tokens_from_symbols(&self, symbols: &[&str]) -> Result<Vec<String>> {
        // Check rate limit
        let limiter = self.rate_limiter.lock().await;
        if !limiter.check(RATE_LIMIT_KEY, RATE_LIMIT).await {
            return Err(Error::RateLimitExceeded("API rate limit exceeded".to_string()));
        }

        let url = format!("{}/cryptocurrency/quotes/latest", API_BASE_URL);
        let response = self.client
            .get(&url)
            .header("X-CMC_PRO_API_KEY", &self.api_key)
            .query(&[("symbol", symbols.join(",")), ("convert", "USD".to_string())])
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError(format!(
                "CoinMarketCap API error: {}",
                response.status()
            )).into());
        }

        let data: serde_json::Value = response.json().await?;
        
        let tokens = data["data"]
            .as_array()
            .ok_or_else(|| ApiError::InvalidFormat("No token data found".to_string()))?
            .iter()
            .filter_map(|token| {
                let symbol = token["symbol"].as_str()?;
                Some(symbol.to_string())
            })
            .collect();

        Ok(tokens)
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<ApiMarketData> {
        let url = format!("{}/cryptocurrency/quotes/latest", self.base_url);
        let params = [
            ("symbol", symbol),
            ("convert", "USD"),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .header("X-CMC_PRO_API_KEY", &self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::ApiError(format!(
                "CoinMarketCap API error: {}",
                response.status()
            )).into());
        }

        let data: CoinMarketCapResponse = response.json().await?;
        
        if data.data.is_empty() {
            return Err(Error::ApiError(
                "No data returned from CoinMarketCap API".to_string()
            ).into());
        }

        let usd_data = &data.data[0].quote.usd;

        Ok(ApiMarketData {
            symbol: symbol.to_string(),
            price: usd_data.price,
            volume: usd_data.volume_24h,
            market_cap: usd_data.market_cap,
            price_change_24h: usd_data.percent_change_24h,
            volume_change_24h: usd_data.volume_change_24h,
            timestamp: Utc::now(),
            volume_24h: usd_data.volume_24h,
            change_24h: usd_data.percent_change_24h,
            quote: Quote {
                usd: crate::api::types::USDData {
                    price: usd_data.price,
                    volume_24h: usd_data.volume_24h,
                    market_cap: usd_data.market_cap,
                    percent_change_24h: usd_data.percent_change_24h,
                    volume_change_24h: usd_data.volume_change_24h,
                }
            }
        })
    }
} 
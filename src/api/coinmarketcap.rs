use crate::api::{MarketData as ApiMarketData, Quote, ApiError};
use crate::error::{Result, Error};
use chrono::{DateTime, Utc};
use log::{info, error};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::security::rate_limit::RateLimiter;
use std::collections::HashMap;

// Define constants locally
const API_BASE_URL: &str = "https://pro-api.coinmarketcap.com/v1";
// const MAX_RETRIES: u32 = 3; // Keep commented
const CACHE_TTL: Duration = Duration::from_secs(60); // Define locally
const RATE_LIMIT_KEY: &str = "coinmarketcap"; // Define locally

#[derive(Debug, Deserialize)]
struct CMCResponse<T> {
    _status: CMCStatus,
    data: Option<T>,
}

#[derive(Debug, Deserialize)]
struct CMCStatus {
    _error_code: u32,
    _error_message: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CMCToken {
    pub _id: u32,
    pub _name: String,
    pub _symbol: String,
    pub _slug: String,
    pub quote: Option<CMCTokenQuote>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CMCTokenQuote {
    #[serde(rename = "USD")]
    pub _usd: Option<CMCTokenQuoteUSD>,
}

// Uncommented struct
#[derive(Debug, Deserialize, Clone)]
pub struct CMCTokenQuoteUSD {
    pub _price: f64,
    pub _volume_24h: f64,
    pub _market_cap: f64,
    pub _percent_change_1h: f64,
    pub _percent_change_24h: f64,
    pub _percent_change_7d: f64,
    pub _last_updated: DateTime<Utc>,
}

// Comment out unused struct
/*
#[derive(Debug, Deserialize, Clone)]
struct CMCTokenList {
    data: Vec<CMCToken>,
}
*/

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarketData {
    pub(crate) price: f64,
    pub(crate) volume: f64,
    pub(crate) market_cap: f64,
    pub(crate) price_change_24h: f64,
    pub(crate) volume_change_24h: f64,
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
    base_url: String,
    api_key: String,
    client: Client,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    _last_request: Instant,
    token_cache: Arc<Mutex<HashMap<String, (CMCToken, Instant)>>>,
    quote_cache: Arc<Mutex<HashMap<String, (CMCTokenQuote, Instant)>>>,
}

impl CoinMarketCapClient {
    pub fn new(api_key: String, rate_limiter: Arc<Mutex<RateLimiter>>) -> Self {
        Self {
            base_url: API_BASE_URL.to_string(),
            api_key,
            client: Client::new(),
            rate_limiter,
            _last_request: Instant::now(),
            token_cache: Arc::new(Mutex::new(HashMap::new())),
            quote_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn check_rate_limit_cmc(&self) -> Result<()> {
        let limiter = self.rate_limiter.lock().await;
        if !limiter.check(RATE_LIMIT_KEY).await? {
            return Err(Error::RateLimitExceeded("CoinMarketCap API rate limit exceeded".to_string()));
        }
        Ok(())
    }

    async fn make_request<T: for<'de> Deserialize<'de>>(
        &mut self,
        endpoint: &str,
        params_opt: Option<HashMap<String, String>>,
    ) -> Result<T> {
        self.check_rate_limit_cmc().await?;

        let mut url = format!("{}{}", self.base_url, endpoint);
        if let Some(ref params) = params_opt {
            if !params.is_empty() {
                url.push('?');
                url.push_str(&serde_urlencoded::to_string(params).map_err(|e| Error::InternalError(e.to_string()))?);
            }
        }

        let request_builder = self.client.get(&url).header("X-CMC_PRO_API_KEY", &self.api_key);

        match request_builder.send().await {
            Ok(response) => {
                match response.status() {
                    StatusCode::OK => response.json::<T>().await.map_err(|e| e.into()),
                    StatusCode::TOO_MANY_REQUESTS => Err(Error::RateLimitExceeded("CoinMarketCap API rate limit exceeded (429)".to_string())),
                    status => {
                        let body = response.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
                        error!("CoinMarketCap API Error: Status {}, Body: {}", status, body);
                        Err(ApiError::RequestError(format!("Request failed with status: {}", status)).into())
                    }
                }
            }
            Err(e) => {
                error!("Reqwest Error for CoinMarketCap: {}", e);
                Err(e.into())
            }
        }
    }

    pub async fn get_token_info(&mut self, token_id: &str) -> Result<CMCToken> {
        let cache_key = token_id.to_string();
        let current_time = Instant::now();
        let cache = self.token_cache.lock().await;
        if let Some((cached_token, timestamp)) = cache.get(&cache_key) {
            if current_time.duration_since(*timestamp) < CACHE_TTL {
                info!("Using cached token info for: {}", token_id);
                return Ok(cached_token.clone());
            }
        }
        drop(cache);

        let params = HashMap::from([("id".to_string(), token_id.to_string())]);
        let response = self.make_request::<CMCResponse<HashMap<String, CMCToken>>>(
            "/cryptocurrency/info", 
            Some(params),
        ).await?;
        
        let token_info = response.data
            .and_then(|mut data| data.remove(token_id))
            .ok_or_else(|| Error::ApiInvalidData(format!("Token info not found for id {}", token_id)))?;
        
        self.token_cache.lock().await.insert(cache_key, (token_info.clone(), Instant::now()));
        
        Ok(token_info)
    }

    pub async fn get_token_quote(&mut self, token_id: &str) -> Result<CMCTokenQuote> {
        let cache_key = token_id.to_string();
        let current_time = Instant::now();
        let cache = self.quote_cache.lock().await;
        if let Some((cached_quote, timestamp)) = cache.get(&cache_key) {
            if current_time.duration_since(*timestamp) < CACHE_TTL {
                info!("Using cached quote for: {}", token_id);
                return Ok(cached_quote.clone());
            }
        }
        drop(cache);

        let params = HashMap::from([("id".to_string(), token_id.to_string())]);
        let response = self.make_request::<CMCResponse<HashMap<String, CMCToken>>>(
            "/cryptocurrency/quotes/latest",
            Some(params),
        ).await?;

        let quote = response.data
            .and_then(|mut data| data.remove(token_id))
            .and_then(|token| token.quote)
            .ok_or_else(|| Error::ApiInvalidData(format!("Quote not found for id {}", token_id)))?;

        self.quote_cache.lock().await.insert(cache_key, (quote.clone(), Instant::now()));
        
        Ok(quote)
    }

    pub async fn get_top_tokens(&mut self, limit: u32) -> Result<Vec<CMCToken>> {
        let endpoint = "/cryptocurrency/listings/latest";
        let params = HashMap::from([
            ("limit".to_string(), limit.to_string()),
            ("sort".to_string(), "market_cap".to_string()),
        ]);
        let response = self.make_request::<CMCResponse<Vec<CMCToken>>>(
            endpoint, 
            Some(params),
        ).await?;
        response.data.ok_or_else(|| Error::ApiInvalidData("No data found for top tokens".to_string()))
    }

    pub async fn get_trending_tokens(&mut self) -> Result<Vec<CMCToken>> {
        let endpoint = "/cryptocurrency/listings/latest";
        let params = HashMap::from([
            ("limit".to_string(), "10".to_string()),
            ("sort".to_string(), "percent_change_24h".to_string()),
        ]);
        let response = self.make_request::<CMCResponse<Vec<CMCToken>>>(
            endpoint, 
            Some(params),
        ).await?;
        response.data.ok_or_else(|| Error::ApiInvalidData("No data found for trending tokens".to_string()))
    }

    pub async fn get_quote(&mut self, symbol: &str) -> Result<CMCTokenQuote> {
        let endpoint = "/cryptocurrency/quotes/latest";
        let params = HashMap::from([("symbol".to_string(), symbol.to_string())]);
        let response = self.make_request::<CMCResponse<HashMap<String, CMCToken>>>(
            endpoint, 
            Some(params),
        ).await?;

        response.data
            .and_then(|mut data| data.remove(symbol))
            .and_then(|token| token.quote)
            .ok_or_else(|| Error::ApiInvalidData(format!("Quote not found for {}", symbol)))
    }

    pub async fn get_quotes(&mut self, symbols: &[&str]) -> Result<HashMap<String, CMCTokenQuote>> {
        let endpoint = "/cryptocurrency/quotes/latest";
        let params = HashMap::from([("symbol".to_string(), symbols.join(",").to_string())]);
        let response = self.make_request::<CMCResponse<HashMap<String, CMCToken>>>(
            endpoint, 
            Some(params),
        ).await?;

        response.data
            .map(|data| {
                data.into_iter()
                    .filter_map(|(symbol, token)| token.quote.map(|quote| (symbol, quote)))
                    .collect()
            })
            .ok_or_else(|| Error::ApiInvalidData("Quotes not found".to_string()))
    }

    pub async fn get_token_quote_from_symbol(&self, symbol: &str) -> Result<Quote> {
        self.check_rate_limit_cmc().await?;

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
            )));
        }

        let data: serde_json::Value = response.json().await?;
        
        let quote = data["data"][symbol]["quote"].clone();
        Ok(serde_json::from_value(quote)?)
    }

    pub async fn get_top_tokens_from_symbols(&self, symbols: &[&str]) -> Result<Vec<String>> {
        self.check_rate_limit_cmc().await?;

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
            )));
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
        self.check_rate_limit_cmc().await?;
        let url = format!("{}/cryptocurrency/quotes/latest", API_BASE_URL);
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
            )));
        }

        let data: CoinMarketCapResponse = response.json().await?;
        
        if data.data.is_empty() {
            return Err(Error::ApiError(
                "No data returned from CoinMarketCap API".to_string()
            ));
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
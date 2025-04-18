use crate::utils::error::Result;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, Instant};
use log::error;

pub mod coingecko;
pub mod coinmarketcap;
pub mod cryptodatadownload;
pub mod types;

pub use coingecko::CoinGeckoClient;
pub use cryptodatadownload::CryptoDataDownloadClient;
pub use types::{MarketData, Quote};

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("API request failed: {0}")]
    RequestError(String),
    #[error("API rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    #[error("API response validation failed: {0}")]
    ValidationError(String),
    #[error("API response format invalid: {0}")]
    InvalidFormat(String),
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    last_request: Arc<Mutex<Instant>>,
    min_interval: Duration,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            last_request: Arc::new(Mutex::new(Instant::now())),
            min_interval: Duration::from_millis(100), // 10 requests per second
        }
    }

    pub async fn wait(&self) {
        let mut last_request = self.last_request.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_request);
        
        if elapsed < self.min_interval {
            tokio::time::sleep(self.min_interval - elapsed).await;
        }
        
        *last_request = Instant::now();
    }

    pub async fn check(&self, _key: &str, interval: Duration) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(*self.last_request.lock().await);
        elapsed >= interval
    }
}

#[derive(Debug, Clone)]
pub struct MarketDataCollector {
    coingecko: Arc<Mutex<coingecko::CoinGeckoClient>>,
    coinmarketcap: Arc<Mutex<coinmarketcap::CoinMarketCapClient>>,
    cryptodatadownload: Arc<Mutex<cryptodatadownload::CryptoDataDownloadClient>>,
}

impl MarketDataCollector {
    pub fn new(
        coingecko_api_key: String,
        coinmarketcap_api_key: String,
        cryptodatadownload_api_key: String,
    ) -> Self {
        Self {
            coingecko: Arc::new(Mutex::new(coingecko::CoinGeckoClient::new(
                Some(coingecko_api_key),
                Arc::new(Mutex::new(RateLimiter::new())),
            ))),
            coinmarketcap: Arc::new(Mutex::new(coinmarketcap::CoinMarketCapClient::new(
                coinmarketcap_api_key,
                Arc::new(Mutex::new(RateLimiter::new())),
            ))),
            cryptodatadownload: Arc::new(Mutex::new(cryptodatadownload::CryptoDataDownloadClient::new(
                cryptodatadownload_api_key,
            ))),
        }
    }

    pub async fn collect_market_data(&self, symbol: &str) -> Result<types::MarketData> {
        // Collect data from all sources
        let coingecko_data: types::MarketData = self.coingecko.lock().await.get_market_data(symbol).await?.into();
        let coinmarketcap_data: types::MarketData = self.coinmarketcap.lock().await.get_market_data(symbol).await?.into();
        let cryptodatadownload_data: types::MarketData = self.cryptodatadownload.lock().await.get_market_data(symbol).await?.into();

        // Calculate weighted averages
        let price = self.calculate_weighted_price(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data);
        let volume = self.calculate_weighted_volume(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data);
        let market_cap = self.calculate_weighted_market_cap(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data);
        let price_change_24h = self.calculate_weighted_price_change(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data);
        let volume_change_24h = self.calculate_weighted_volume_change(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data);

        Ok(types::MarketData {
            symbol: symbol.to_string(),
            price,
            volume,
            market_cap,
            price_change_24h,
            volume_change_24h,
            timestamp: Utc::now(),
            volume_24h: volume,
            change_24h: price_change_24h,
            quote: types::Quote {
                usd: types::USDData {
                    price,
                    volume_24h: volume,
                    market_cap,
                    percent_change_24h: price_change_24h,
                    volume_change_24h,
                }
            }
        })
    }

    fn calculate_weighted_price(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        // Weighted average based on market cap
        let total_market_cap = coingecko.market_cap + coinmarketcap.market_cap + cryptodatadownload.market_cap;
        if total_market_cap == 0.0 {
            return (coingecko.price + coinmarketcap.price + cryptodatadownload.price) / 3.0;
        }

        (coingecko.price * coingecko.market_cap + 
         coinmarketcap.price * coinmarketcap.market_cap + 
         cryptodatadownload.price * cryptodatadownload.market_cap) / total_market_cap
    }

    fn calculate_weighted_volume(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        // Simple average since volume is more volatile
        (coingecko.volume + coinmarketcap.volume + cryptodatadownload.volume) / 3.0
    }

    fn calculate_weighted_market_cap(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        // Simple average since market cap is more stable
        (coingecko.market_cap + coinmarketcap.market_cap + cryptodatadownload.market_cap) / 3.0
    }

    fn calculate_weighted_price_change(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        // Simple average since price changes are more volatile
        (coingecko.price_change_24h + coinmarketcap.price_change_24h + cryptodatadownload.price_change_24h) / 3.0
    }

    fn calculate_weighted_volume_change(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        // Simple average since volume changes are more volatile
        (coingecko.volume_change_24h + coinmarketcap.volume_change_24h + cryptodatadownload.volume_change_24h) / 3.0
    }
} 
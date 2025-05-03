use crate::error::Result;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Duration;
use log::error;
use crate::security::rate_limit::RateLimiter as SecurityRateLimiter;
use async_trait::async_trait;

pub mod coingecko;
pub mod coinmarketcap;
pub mod cryptodatadownload;
pub mod types;

pub use coingecko::CoinGeckoClient;
pub use cryptodatadownload::CryptoDataDownloadClient;
pub use types::{MarketData, Quote};

#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    async fn collect_market_data(&mut self, symbol: &str) -> Result<types::MarketData>;
}

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
        let coingecko_limiter = Arc::new(Mutex::new(SecurityRateLimiter::new(10, Duration::from_secs(1))));
        let coinmarketcap_limiter = Arc::new(Mutex::new(SecurityRateLimiter::new(10, Duration::from_secs(1))));

        Self {
            coingecko: Arc::new(Mutex::new(coingecko::CoinGeckoClient::new(
                Some(coingecko_api_key),
                coingecko_limiter,
            ))),
            coinmarketcap: Arc::new(Mutex::new(coinmarketcap::CoinMarketCapClient::new(
                coinmarketcap_api_key,
                coinmarketcap_limiter,
            ))),
            cryptodatadownload: Arc::new(Mutex::new(cryptodatadownload::CryptoDataDownloadClient::new(
                cryptodatadownload_api_key,
            ))),
        }
    }

    fn calculate_weighted_price(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
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
        (coingecko.volume + coinmarketcap.volume + cryptodatadownload.volume) / 3.0
    }

    fn calculate_weighted_market_cap(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        (coingecko.market_cap + coinmarketcap.market_cap + cryptodatadownload.market_cap) / 3.0
    }

    fn calculate_weighted_price_change(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        (coingecko.price_change_24h + coinmarketcap.price_change_24h + cryptodatadownload.price_change_24h) / 3.0
    }

    fn calculate_weighted_volume_change(
        &self,
        coingecko: &types::MarketData,
        coinmarketcap: &types::MarketData,
        cryptodatadownload: &types::MarketData,
    ) -> f64 {
        (coingecko.volume_change_24h + coinmarketcap.volume_change_24h + cryptodatadownload.volume_change_24h) / 3.0
    }
}

#[async_trait]
impl MarketDataProvider for MarketDataCollector {
    async fn collect_market_data(&mut self, symbol: &str) -> Result<types::MarketData> {
        let coingecko_guard = self.coingecko.lock().await;
        let coinmarketcap_guard = self.coinmarketcap.lock().await;
        let cryptodatadownload_guard = self.cryptodatadownload.lock().await;

        let coingecko_data = coingecko_guard.get_market_data(symbol).await?;
        let coinmarketcap_data = coinmarketcap_guard.get_market_data(symbol).await?;
        let cryptodatadownload_data: types::MarketData = cryptodatadownload_guard.get_market_data(symbol).await?.into();

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
} 
use crate::utils::error::Result;
use crate::trading::MarketData;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use reqwest::Client;
use log::{info, error};
use crate::web::validation::RateLimiter;

pub mod coingecko;
pub mod coinmarketcap;
pub mod cryptodatadownload;

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
        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new()));
        
        Self {
            coingecko: Arc::new(Mutex::new(coingecko::CoinGeckoClient::new(
                coingecko_api_key,
                rate_limiter.clone(),
            ))),
            coinmarketcap: Arc::new(Mutex::new(coinmarketcap::CoinMarketCapClient::new(
                coinmarketcap_api_key,
                rate_limiter.clone(),
            ))),
            cryptodatadownload: Arc::new(Mutex::new(cryptodatadownload::CryptoDataDownloadClient::new(
                cryptodatadownload_api_key,
            ))),
        }
    }

    pub async fn collect_market_data(&self, symbol: &str) -> Result<MarketData> {
        // Collect data from all sources
        let coingecko_data = self.coingecko.lock().await.get_market_data(symbol).await?;
        let coinmarketcap_data = self.coinmarketcap.lock().await.get_market_data(symbol).await?;
        let cryptodatadownload_data = self.cryptodatadownload.lock().await.get_market_data(symbol).await?;

        // Combine and validate data
        let market_data = MarketData {
            timestamp: Utc::now(),
            symbol: symbol.to_string(),
            price: self.calculate_weighted_price(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data),
            volume: self.calculate_weighted_volume(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data),
            market_cap: self.calculate_weighted_market_cap(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data),
            price_change_24h: self.calculate_weighted_price_change(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data),
            volume_change_24h: self.calculate_weighted_volume_change(&coingecko_data, &coinmarketcap_data, &cryptodatadownload_data),
        };

        Ok(market_data)
    }

    fn calculate_weighted_price(
        &self,
        coingecko: &coingecko::MarketData,
        coinmarketcap: &coinmarketcap::MarketData,
        cryptodatadownload: &cryptodatadownload::MarketData,
    ) -> f64 {
        // Weight the prices based on reliability and volume
        let weights = vec![0.4, 0.4, 0.2]; // Adjust weights based on reliability
        let prices = vec![coingecko.price, coinmarketcap.price, cryptodatadownload.price];
        
        prices.iter().zip(weights.iter())
            .map(|(price, weight)| price * weight)
            .sum()
    }

    fn calculate_weighted_volume(
        &self,
        coingecko: &coingecko::MarketData,
        coinmarketcap: &coinmarketcap::MarketData,
        cryptodatadownload: &cryptodatadownload::MarketData,
    ) -> f64 {
        // Use the highest volume as it's likely the most accurate
        vec![coingecko.volume, coinmarketcap.volume, cryptodatadownload.volume]
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    fn calculate_weighted_market_cap(
        &self,
        coingecko: &coingecko::MarketData,
        coinmarketcap: &coinmarketcap::MarketData,
        cryptodatadownload: &cryptodatadownload::MarketData,
    ) -> f64 {
        // Use CoinGecko and CoinMarketCap data primarily for market cap
        let weights = vec![0.5, 0.5, 0.0];
        let market_caps = vec![coingecko.market_cap, coinmarketcap.market_cap, cryptodatadownload.market_cap];
        
        market_caps.iter().zip(weights.iter())
            .map(|(cap, weight)| cap * weight)
            .sum()
    }

    fn calculate_weighted_price_change(
        &self,
        coingecko: &coingecko::MarketData,
        coinmarketcap: &coinmarketcap::MarketData,
        cryptodatadownload: &cryptodatadownload::MarketData,
    ) -> f64 {
        let weights = vec![0.4, 0.4, 0.2];
        let changes = vec![coingecko.price_change_24h, coinmarketcap.price_change_24h, cryptodatadownload.price_change_24h];
        
        changes.iter().zip(weights.iter())
            .map(|(change, weight)| change * weight)
            .sum()
    }

    fn calculate_weighted_volume_change(
        &self,
        coingecko: &coingecko::MarketData,
        coinmarketcap: &coinmarketcap::MarketData,
        cryptodatadownload: &cryptodatadownload::MarketData,
    ) -> f64 {
        let weights = vec![0.4, 0.4, 0.2];
        let changes = vec![coingecko.volume_change_24h, coinmarketcap.volume_change_24h, cryptodatadownload.volume_change_24h];
        
        changes.iter().zip(weights.iter())
            .map(|(change, weight)| change * weight)
            .sum()
    }
} 
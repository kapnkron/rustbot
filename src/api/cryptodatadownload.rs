use crate::utils::error::Result;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CryptoDataDownloadResponse {
    data: Vec<CryptoDataDownloadData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CryptoDataDownloadData {
    close: f64,
    volume: f64,
    market_cap: f64,
    price_change_24h: f64,
    volume_change_24h: f64,
}

pub struct CryptoDataDownloadClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl CryptoDataDownloadClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
            api_key,
            base_url: "https://api.cryptodatadownload.com/v1".to_string(),
        }
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<MarketData> {
        let url = format!("{}/data/histoday", self.base_url);
        let params = [
            ("fsym", symbol),
            ("tsym", "USD"),
            ("limit", "1"),
            ("api_key", &self.api_key),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(crate::utils::error::TradingError::ApiError(format!(
                "CryptoDataDownload API error: {}",
                response.status()
            )).into());
        }

        let data: CryptoDataDownloadResponse = response.json().await?;
        
        if data.data.is_empty() {
            return Err(crate::utils::error::TradingError::ApiError(
                "No data returned from CryptoDataDownload API".to_string()
            ).into());
        }

        let latest_data = &data.data[0];

        Ok(MarketData {
            price: latest_data.close,
            volume: latest_data.volume,
            market_cap: latest_data.market_cap,
            price_change_24h: latest_data.price_change_24h,
            volume_change_24h: latest_data.volume_change_24h,
        })
    }
} 
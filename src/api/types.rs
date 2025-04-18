use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::models::market::TokenData;
use crate::utils::error::{Result, Error};
use log::{error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
    pub timestamp: DateTime<Utc>,
    pub volume_24h: f64,
    pub change_24h: f64,
    pub quote: Quote,
}

impl TryFrom<MarketData> for TokenData {
    type Error = Error;

    fn try_from(market_data: MarketData) -> Result<TokenData> {
        let now = Utc::now();
        info!("Starting MarketData to TokenData conversion for symbol: {} at {}", 
            market_data.symbol, now.format("%Y-%m-%d %H:%M:%S%.3f"));

        // Log input data for debugging
        info!("Input MarketData - Price: {:.8}, Volume: {:.2}, Market Cap: {:.2}, Timestamp: {}", 
            market_data.price, 
            market_data.volume_24h, 
            market_data.market_cap,
            market_data.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"));

        // Validate required fields
        if market_data.symbol.is_empty() {
            let err_msg = "Empty symbol in MarketData";
            error!("{} - Symbol: '{}', Timestamp: {}", 
                err_msg, 
                market_data.symbol, 
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg.to_string()));
        }
        if !market_data.price.is_finite() || market_data.price <= 0.0 {
            let err_msg = format!("Invalid price in MarketData: {}", market_data.price);
            error!("{} - Symbol: {}, Current Price: {:.8}, Timestamp: {}", 
                err_msg, 
                market_data.symbol,
                market_data.price,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }
        if !market_data.volume_24h.is_finite() || market_data.volume_24h < 0.0 {
            let err_msg = format!("Invalid volume in MarketData: {}", market_data.volume_24h);
            error!("{} - Symbol: {}, Current Volume: {:.2}, Timestamp: {}", 
                err_msg, 
                market_data.symbol,
                market_data.volume_24h,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }
        if !market_data.market_cap.is_finite() || market_data.market_cap < 0.0 {
            let err_msg = format!("Invalid market cap in MarketData: {}", market_data.market_cap);
            error!("{} - Symbol: {}, Current Market Cap: {:.2}, Timestamp: {}", 
                err_msg, 
                market_data.symbol,
                market_data.market_cap,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }

        // Log successful conversion with detailed output
        let result = TokenData {
            address: market_data.symbol.clone(),
            symbol: market_data.symbol.clone(),
            name: "".to_string(),
            price: market_data.price,
            volume_24h: market_data.volume_24h,
            liquidity: 0.0,
            market_cap: market_data.market_cap,
            holders: 0,
            transactions_24h: 0,
            last_updated: market_data.timestamp,
        };

        info!("Successfully converted MarketData to TokenData - Symbol: {}, Price: {:.8}, Volume: {:.2}, Market Cap: {:.2}, Timestamp: {}", 
            result.symbol,
            result.price,
            result.volume_24h,
            result.market_cap,
            result.last_updated.format("%Y-%m-%d %H:%M:%S%.3f"));

        Ok(result)
    }
}

impl TryFrom<TokenData> for MarketData {
    type Error = Error;

    fn try_from(token_data: TokenData) -> Result<MarketData> {
        let now = Utc::now();
        info!("Starting TokenData to MarketData conversion for symbol: {} at {}", 
            token_data.symbol, 
            now.format("%Y-%m-%d %H:%M:%S%.3f"));

        // Log input data for debugging
        info!("Input TokenData - Price: {:.8}, Volume: {:.2}, Market Cap: {:.2}, Timestamp: {}", 
            token_data.price, 
            token_data.volume_24h, 
            token_data.market_cap,
            token_data.last_updated.format("%Y-%m-%d %H:%M:%S%.3f"));

        // Validate required fields
        if token_data.symbol.is_empty() {
            let err_msg = "Empty symbol in TokenData";
            error!("{} - Symbol: '{}', Timestamp: {}", 
                err_msg, 
                token_data.symbol, 
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg.to_string()));
        }
        if !token_data.price.is_finite() || token_data.price <= 0.0 {
            let err_msg = format!("Invalid price in TokenData: {}", token_data.price);
            error!("{} - Symbol: {}, Current Price: {:.8}, Timestamp: {}", 
                err_msg, 
                token_data.symbol,
                token_data.price,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }
        if !token_data.volume_24h.is_finite() || token_data.volume_24h < 0.0 {
            let err_msg = format!("Invalid volume in TokenData: {}", token_data.volume_24h);
            error!("{} - Symbol: {}, Current Volume: {:.2}, Timestamp: {}", 
                err_msg, 
                token_data.symbol,
                token_data.volume_24h,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }
        if !token_data.market_cap.is_finite() || token_data.market_cap < 0.0 {
            let err_msg = format!("Invalid market cap in TokenData: {}", token_data.market_cap);
            error!("{} - Symbol: {}, Current Market Cap: {:.2}, Timestamp: {}", 
                err_msg, 
                token_data.symbol,
                token_data.market_cap,
                now.format("%Y-%m-%d %H:%M:%S%.3f"));
            return Err(Error::ApiInvalidData(err_msg));
        }

        // Log successful conversion with detailed output
        let result = MarketData {
            symbol: token_data.symbol.clone(),
            price: token_data.price,
            volume: token_data.volume_24h,
            market_cap: token_data.market_cap,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
            timestamp: token_data.last_updated,
            volume_24h: token_data.volume_24h,
            change_24h: 0.0,
            quote: Quote {
                usd: USDData {
                    price: token_data.price,
                    volume_24h: token_data.volume_24h,
                    market_cap: token_data.market_cap,
                    percent_change_24h: 0.0,
                    volume_change_24h: 0.0,
                }
            }
        };

        info!("Successfully converted TokenData to MarketData - Symbol: {}, Price: {:.8}, Volume: {:.2}, Market Cap: {:.2}, Timestamp: {}", 
            result.symbol,
            result.price,
            result.volume_24h,
            result.market_cap,
            result.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"));

        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    #[serde(rename = "USD")]
    pub usd: USDData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct USDData {
    pub price: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
    pub percent_change_24h: f64,
    pub volume_change_24h: f64,
} 
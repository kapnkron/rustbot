use crate::api::types::MarketData;
use crate::error::Result;

pub trait TradingStrategy {
    fn analyze(&self, market_data: &MarketData) -> Result<bool>;
    fn get_name(&self) -> &'static str;
}

#[derive(Debug)]
pub struct SimpleMovingAverage {
    // Comment out unused fields
    // short_window: usize,
    // long_window: usize,
}

impl SimpleMovingAverage {
    pub fn new(_short_window: usize, _long_window: usize) -> Self {
        Self { 
            // short_window, long_window 
        }
    }
}

impl TradingStrategy for SimpleMovingAverage {
    fn analyze(&self, _market_data: &MarketData) -> Result<bool> {
        // Implement analysis logic here
        Ok(true)
    }

    fn get_name(&self) -> &'static str {
        "Simple Moving Average"
    }
} 
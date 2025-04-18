use crate::api::types::MarketData;
use crate::error::Result;

pub trait TradingStrategy {
    fn analyze(&self, market_data: &MarketData) -> Result<bool>;
    fn get_name(&self) -> &'static str;
}

pub struct SimpleMovingAverage {
    short_window: usize,
    long_window: usize,
}

impl SimpleMovingAverage {
    pub fn new(short_window: usize, long_window: usize) -> Self {
        Self {
            short_window,
            long_window,
        }
    }
}

impl TradingStrategy for SimpleMovingAverage {
    fn analyze(&self, market_data: &MarketData) -> Result<bool> {
        // TODO: Implement SMA strategy
        Ok(false)
    }

    fn get_name(&self) -> &'static str {
        "Simple Moving Average"
    }
} 
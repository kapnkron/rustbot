use crate::error::Result;
use crate::trading::TradingMarketData;
use crate::ml::MLConfigError;
use std::collections::VecDeque;

pub struct DataPreprocessor {
    window_size: usize,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    min_data_points: usize,
}

impl DataPreprocessor {
    pub fn new(window_size: usize, min_data_points: usize) -> Result<Self> {
        if window_size < 2 {
            return Err(MLConfigError::InvalidConfig("Window size must be at least 2".to_string()).into());
        }
        if min_data_points < window_size {
            return Err(MLConfigError::InvalidConfig("Minimum data points must be at least window size".to_string()).into());
        }

        Ok(Self {
            window_size,
            price_history: VecDeque::with_capacity(window_size),
            volume_history: VecDeque::with_capacity(window_size),
            min_data_points,
        })
    }

    pub fn process_market_data(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
        // Update history
        self.price_history.push_back(data.price);
        self.volume_history.push_back(data.volume);
        
        if self.price_history.len() > self.window_size {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }

        // Check if we have enough data
        if self.price_history.len() < self.min_data_points {
            return Err(MLConfigError::InvalidConfig(format!(
                "Insufficient data points: got {}, required {}",
                self.price_history.len(),
                self.min_data_points
            )).into());
        }

        // Calculate technical indicators
        let rsi = self.calculate_rsi()?;
        let sma = self.calculate_sma()?;
        let ema = self.calculate_ema()?;
        let volume_sma = self.calculate_volume_sma()?;
        let price_volatility = self.calculate_price_volatility()?;
        let macd = self.calculate_macd()?;

        // Normalize features
        let normalized_price = self.normalize_price(data.price)?;
        let normalized_volume = self.normalize_volume(data.volume)?;
        let normalized_market_cap = self.normalize_market_cap(data.market_cap)?;

        // Combine all features
        Ok(vec![
            normalized_price,
            normalized_volume,
            normalized_market_cap,
            rsi,
            sma,
            ema,
            volume_sma,
            price_volatility,
            macd,
            data.price_change_24h,
            data.volume_change_24h,
        ])
    }

    fn calculate_rsi(&self) -> Result<f64> {
        if self.price_history.len() < 2 {
            return Err(MLConfigError::InvalidConfig("Insufficient data for RSI calculation".to_string()).into());
        }

        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;

        for i in 1..self.price_history.len() {
            let change = self.price_history[i] - self.price_history[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
            count += 1;
        }

        if count == 0 {
            return Ok(50.0);
        }

        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);
        }

        let rs = avg_gain / avg_loss;
        Ok(100.0 - (100.0 / (1.0 + rs)))
    }

    fn calculate_sma(&self) -> Result<f64> {
        if self.price_history.is_empty() {
            return Err(MLConfigError::InvalidConfig("No price history available".to_string()).into());
        }
        Ok(self.price_history.iter().sum::<f64>() / self.price_history.len() as f64)
    }

    fn calculate_ema(&self) -> Result<f64> {
        if self.price_history.is_empty() {
            return Err(MLConfigError::InvalidConfig("No price history available".to_string()).into());
        }

        let smoothing = 2.0 / (self.price_history.len() as f64 + 1.0);
        let mut ema = self.price_history[0];

        for &price in self.price_history.iter().skip(1) {
            ema = (price - ema) * smoothing + ema;
        }

        Ok(ema)
    }

    fn calculate_macd(&self) -> Result<f64> {
        if self.price_history.len() < 26 {
            return Err(MLConfigError::InvalidConfig("Insufficient data for MACD calculation".to_string()).into());
        }

        let ema12 = self.calculate_ema_with_period(12)?;
        let ema26 = self.calculate_ema_with_period(26)?;
        Ok(ema12 - ema26)
    }

    fn calculate_ema_with_period(&self, period: usize) -> Result<f64> {
        if self.price_history.len() < period {
            return Err(MLConfigError::InvalidConfig(format!(
                "Insufficient data for {}-period EMA calculation",
                period
            )).into());
        }

        let smoothing = 2.0 / (period as f64 + 1.0);
        let mut ema = self.price_history[0];

        for &price in self.price_history.iter().take(period).skip(1) {
            ema = (price - ema) * smoothing + ema;
        }

        Ok(ema)
    }

    fn calculate_volume_sma(&self) -> Result<f64> {
        if self.volume_history.is_empty() {
            return Err(MLConfigError::InvalidConfig("No volume history available".to_string()).into());
        }
        Ok(self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64)
    }

    fn calculate_price_volatility(&self) -> Result<f64> {
        if self.price_history.len() < 2 {
            return Err(MLConfigError::InvalidConfig("Insufficient data for volatility calculation".to_string()).into());
        }

        let mean = self.calculate_sma()?;
        let variance = self.price_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.price_history.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }

    fn normalize_price(&self, price: f64) -> Result<f64> {
        if self.price_history.is_empty() {
            return Err(MLConfigError::InvalidConfig("No price history available".to_string()).into());
        }
        let min = self.price_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.price_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max == min {
            return Ok(0.5); // Return middle value if all prices are the same
        }
        Ok((price - min) / (max - min))
    }

    fn normalize_volume(&self, volume: f64) -> Result<f64> {
        if self.volume_history.is_empty() {
            return Err(MLConfigError::InvalidConfig("No volume history available".to_string()).into());
        }
        let min = self.volume_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.volume_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if max == min {
            return Ok(0.5); // Return middle value if all volumes are the same
        }
        Ok((volume - min) / (max - min))
    }

    fn normalize_market_cap(&self, market_cap: f64) -> Result<f64> {
        // Log normalization with clipping to avoid extreme values
        let normalized = (market_cap.ln() - 20.0) / 10.0;
        Ok(normalized.max(0.0).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_market_data(price: f64, volume: f64, market_cap: f64) -> TradingMarketData {
        TradingMarketData {
            symbol: "BTC".to_string(),
            price,
            volume,
            market_cap,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
            timestamp: Utc::now(),
            volume_24h: volume,
            change_24h: 0.0,
            quote: crate::api::types::Quote {
                usd: crate::api::types::USDData {
                    price,
                    volume_24h: volume,
                    market_cap,
                    percent_change_24h: 0.0,
                    volume_change_24h: 0.0,
                }
            }
        }
    }

    #[test]
    fn test_preprocessing() -> Result<()> {
        let mut preprocessor = DataPreprocessor::new(10, 5)?;
        
        // Test with increasing prices
        let data = create_test_market_data(100.0, 1000.0, 1_000_000_000.0);
        let features = preprocessor.process_market_data(&data)?;
        assert_eq!(features.len(), 11);
        
        // Test RSI calculation
        let data2 = create_test_market_data(110.0, 1100.0, 1_100_000_000.0);
        preprocessor.process_market_data(&data2)?;
        let rsi = preprocessor.calculate_rsi()?;
        assert!(rsi > 50.0); // Should be bullish
        
        // Test normalization
        let normalized_price = preprocessor.normalize_price(110.0)?;
        assert!(normalized_price >= 0.0 && normalized_price <= 1.0);
        
        Ok(())
    }
} 
use crate::api::MarketData;
use crate::error::Result;

pub struct DataPreprocessor {
    window_size: usize,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
}

impl DataPreprocessor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            price_history: Vec::with_capacity(window_size),
            volume_history: Vec::with_capacity(window_size),
        }
    }

    pub fn process_market_data(&mut self, data: &MarketData) -> Result<Vec<f64>> {
        // Update history
        self.price_history.push(data.price);
        self.volume_history.push(data.volume);
        
        if self.price_history.len() > self.window_size {
            self.price_history.remove(0);
            self.volume_history.remove(0);
        }

        // Calculate technical indicators
        let rsi = self.calculate_rsi();
        let sma = self.calculate_sma();
        let volume_sma = self.calculate_volume_sma();
        let price_volatility = self.calculate_price_volatility();

        // Normalize features
        let normalized_price = self.normalize_price(data.price);
        let normalized_volume = self.normalize_volume(data.volume);
        let normalized_market_cap = self.normalize_market_cap(data.market_cap);

        // Combine all features
        Ok(vec![
            normalized_price,
            normalized_volume,
            normalized_market_cap,
            rsi,
            sma,
            volume_sma,
            price_volatility,
            data.price_change_24h,
            data.volume_change_24h,
        ])
    }

    fn calculate_rsi(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 50.0;
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
            return 50.0;
        }

        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_sma(&self) -> f64 {
        if self.price_history.is_empty() {
            return 0.0;
        }
        self.price_history.iter().sum::<f64>() / self.price_history.len() as f64
    }

    fn calculate_volume_sma(&self) -> f64 {
        if self.volume_history.is_empty() {
            return 0.0;
        }
        self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64
    }

    fn calculate_price_volatility(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }

        let mean = self.calculate_sma();
        let variance = self.price_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.price_history.len() - 1) as f64;
        
        variance.sqrt()
    }

    fn normalize_price(&self, price: f64) -> f64 {
        if self.price_history.is_empty() {
            return 0.0;
        }
        let min = self.price_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.price_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (price - min) / (max - min)
    }

    fn normalize_volume(&self, volume: f64) -> f64 {
        if self.volume_history.is_empty() {
            return 0.0;
        }
        let min = self.volume_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.volume_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (volume - min) / (max - min)
    }

    fn normalize_market_cap(&self, market_cap: f64) -> f64 {
        // Simple log normalization for market cap
        (market_cap.ln() - 20.0) / 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_market_data(price: f64, volume: f64, market_cap: f64) -> MarketData {
        MarketData {
            timestamp: Utc::now(),
            symbol: "BTC".to_string(),
            price,
            volume,
            market_cap,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
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
    fn test_preprocessing() {
        let mut preprocessor = DataPreprocessor::new(10);
        
        // Test with increasing prices
        let data = create_test_market_data(100.0, 1000.0, 1_000_000_000.0);
        let features = preprocessor.process_market_data(&data).unwrap();
        assert_eq!(features.len(), 9);
        
        // Test RSI calculation
        let data2 = create_test_market_data(110.0, 1100.0, 1_100_000_000.0);
        preprocessor.process_market_data(&data2).unwrap();
        let rsi = preprocessor.calculate_rsi();
        assert!(rsi > 50.0); // Should be bullish
        
        // Test normalization
        let normalized_price = preprocessor.normalize_price(110.0);
        assert!(normalized_price >= 0.0 && normalized_price <= 1.0);
    }
} 
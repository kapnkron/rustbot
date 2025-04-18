use crate::api::MarketData;
use crate::error::Result;
use std::collections::VecDeque;

pub struct FeatureExtractor {
    window_size: usize,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    market_cap_history: VecDeque<f64>,
}

impl FeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            price_history: VecDeque::with_capacity(window_size),
            volume_history: VecDeque::with_capacity(window_size),
            market_cap_history: VecDeque::with_capacity(window_size),
        }
    }

    pub fn update(&mut self, data: &MarketData) {
        self.price_history.push_back(data.price);
        self.volume_history.push_back(data.volume);
        self.market_cap_history.push_back(data.market_cap);

        if self.price_history.len() > self.window_size {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.market_cap_history.pop_front();
        }
    }

    pub fn extract_features(&self, data: &MarketData) -> Result<Vec<f64>> {
        if self.price_history.len() < self.window_size {
            return Err("Insufficient historical data".into());
        }

        // Price-based features
        let price_features = self.extract_price_features(data.price);
        
        // Volume-based features
        let volume_features = self.extract_volume_features(data.volume);
        
        // Market cap features
        let market_cap_features = self.extract_market_cap_features(data.market_cap);
        
        // Technical indicators
        let technical_features = self.extract_technical_indicators();
        
        // Combine all features
        let mut features = Vec::new();
        features.extend(price_features);
        features.extend(volume_features);
        features.extend(market_cap_features);
        features.extend(technical_features);
        
        // Add raw features
        features.push(data.price_change_24h);
        features.push(data.volume_change_24h);

        Ok(features)
    }

    fn extract_price_features(&self, current_price: f64) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Price momentum
        let price_momentum = (current_price - self.price_history.front().unwrap()) / self.price_history.front().unwrap();
        features.push(price_momentum);
        
        // Price volatility
        let mean_price = self.price_history.iter().sum::<f64>() / self.price_history.len() as f64;
        let variance = self.price_history.iter()
            .map(|&x| (x - mean_price).powi(2))
            .sum::<f64>() / self.price_history.len() as f64;
        features.push(variance.sqrt());
        
        // Price range
        let max_price = self.price_history.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min_price = self.price_history.iter().fold(f64::MAX, |a, &b| a.min(b));
        features.push((max_price - min_price) / mean_price);
        
        features
    }

    fn extract_volume_features(&self, current_volume: f64) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Volume momentum
        let volume_momentum = (current_volume - self.volume_history.front().unwrap()) / self.volume_history.front().unwrap();
        features.push(volume_momentum);
        
        // Volume volatility
        let mean_volume = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
        let variance = self.volume_history.iter()
            .map(|&x| (x - mean_volume).powi(2))
            .sum::<f64>() / self.volume_history.len() as f64;
        features.push(variance.sqrt());
        
        // Volume-price correlation
        let price_mean = self.price_history.iter().sum::<f64>() / self.price_history.len() as f64;
        let volume_mean = mean_volume;
        let covariance = self.price_history.iter().zip(self.volume_history.iter())
            .map(|(&p, &v)| (p - price_mean) * (v - volume_mean))
            .sum::<f64>() / self.price_history.len() as f64;
        let price_std = self.price_history.iter()
            .map(|&x| (x - price_mean).powi(2))
            .sum::<f64>().sqrt() / self.price_history.len() as f64;
        let volume_std = variance.sqrt();
        features.push(covariance / (price_std * volume_std));
        
        features
    }

    fn extract_market_cap_features(&self, current_market_cap: f64) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Market cap momentum
        let market_cap_momentum = (current_market_cap - self.market_cap_history.front().unwrap()) / self.market_cap_history.front().unwrap();
        features.push(market_cap_momentum);
        
        // Market cap volatility
        let mean_market_cap = self.market_cap_history.iter().sum::<f64>() / self.market_cap_history.len() as f64;
        let variance = self.market_cap_history.iter()
            .map(|&x| (x - mean_market_cap).powi(2))
            .sum::<f64>() / self.market_cap_history.len() as f64;
        features.push(variance.sqrt());
        
        features
    }

    fn extract_technical_indicators(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        // RSI
        features.push(self.calculate_rsi());
        
        // Moving averages
        let (sma, ema) = self.calculate_moving_averages();
        features.push(sma);
        features.push(ema);
        
        // MACD
        let (macd, signal) = self.calculate_macd();
        features.push(macd);
        features.push(signal);
        
        features
    }

    fn calculate_rsi(&self) -> f64 {
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..self.price_history.len() {
            let change = self.price_history[i] - self.price_history[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / self.price_history.len() as f64;
        let avg_loss = losses / self.price_history.len() as f64;
        
        if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    }

    fn calculate_moving_averages(&self) -> (f64, f64) {
        // Simple Moving Average
        let sma = self.price_history.iter().sum::<f64>() / self.price_history.len() as f64;
        
        // Exponential Moving Average
        let multiplier = 2.0 / (self.price_history.len() as f64 + 1.0);
        let mut ema = self.price_history[0];
        
        for &price in self.price_history.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }
        
        (sma, ema)
    }

    fn calculate_macd(&self) -> (f64, f64) {
        // Calculate 12-period EMA
        let ema12 = self.calculate_ema(12);
        
        // Calculate 26-period EMA
        let ema26 = self.calculate_ema(26);
        
        // Calculate MACD line
        let macd = ema12 - ema26;
        
        // Calculate signal line (9-period EMA of MACD)
        let signal = self.calculate_ema_of_ema(9);
        
        (macd, signal)
    }

    fn calculate_ema(&self, period: usize) -> f64 {
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = self.price_history[0];
        
        for &price in self.price_history.iter().take(period) {
            ema = (price - ema) * multiplier + ema;
        }
        
        ema
    }

    fn calculate_ema_of_ema(&self, period: usize) -> f64 {
        let mut macd_values = Vec::new();
        let mut ema12 = self.price_history[0];
        let mut ema26 = self.price_history[0];
        
        let multiplier12 = 2.0 / (12.0 + 1.0);
        let multiplier26 = 2.0 / (26.0 + 1.0);
        
        for &price in self.price_history.iter() {
            ema12 = (price - ema12) * multiplier12 + ema12;
            ema26 = (price - ema26) * multiplier26 + ema26;
            macd_values.push(ema12 - ema26);
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut signal = macd_values[0];
        
        for &macd in macd_values.iter().take(period) {
            signal = (macd - signal) * multiplier + signal;
        }
        
        signal
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
        }
    }

    #[test]
    fn test_feature_extraction() -> Result<()> {
        let mut extractor = FeatureExtractor::new(30);
        
        // Create test data
        for i in 0..30 {
            let price = 100.0 + (i as f64 * 0.1);
            let volume = 1000.0 + (i as f64 * 10.0);
            let market_cap = 1_000_000_000.0 + (i as f64 * 10_000_000.0);
            
            let data = create_test_market_data(price, volume, market_cap);
            extractor.update(&data);
        }
        
        // Test feature extraction
        let data = create_test_market_data(103.0, 1300.0, 1_300_000_000.0);
        let features = extractor.extract_features(&data)?;
        
        assert!(!features.is_empty());
        assert!(features.iter().all(|&x| !x.is_nan()));
        assert!(features.iter().all(|&x| !x.is_infinite()));
        
        Ok(())
    }
} 
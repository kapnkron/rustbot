use crate::error::Result;
use crate::trading::TradingMarketData;
use crate::ml::MLConfigError;
use std::collections::VecDeque;
use crate::api::MarketData;
use log::debug;

#[derive(Debug)]
pub struct DataPreprocessor {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    min_data_points: usize,
}

impl DataPreprocessor {
    pub fn new(min_data_points: usize) -> Result<Self> {
        if min_data_points < 26 {
            return Err(MLConfigError::InvalidConfig(format!(
                "Minimum data points must be at least 26 for MACD, got {}",
                min_data_points
            )).into());
        }

        Ok(Self {
            price_history: VecDeque::with_capacity(min_data_points),
            volume_history: VecDeque::with_capacity(min_data_points),
            min_data_points,
        })
    }

    pub fn process_market_data(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
        self.price_history.push_back(data.price);
        self.volume_history.push_back(data.volume);
        
        if self.price_history.len() > self.min_data_points {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }

        if self.price_history.len() < self.min_data_points {
            return Err(MLConfigError::InvalidConfig(format!(
                "Insufficient data points: got {}, required {}",
                self.price_history.len(),
                self.min_data_points
            )).into());
        }

        let rsi = self.calculate_rsi()?;
        let sma = self.calculate_sma()?;
        let ema = self.calculate_ema()?;
        let volume_sma = self.calculate_volume_sma()?;
        let price_volatility = self.calculate_price_volatility()?;
        let macd = self.calculate_macd()?;

        let normalized_price = self.normalize_price(data.price)?;
        let normalized_volume = self.normalize_volume(data.volume)?;
        let normalized_market_cap = self.normalize_market_cap(data.market_cap)?;

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
        Ok(normalized.clamp(0.0, 1.0))
    }
}

/// Prepares feature vectors and corresponding classification labels from historical data.
/// 
/// Args:
/// * `historical_data`: Slice of raw market data points, ordered chronologically.
/// * `min_history_points`: Minimum data points needed by the preprocessor (e.g., 26 for MACD).
/// * `prediction_horizon`: How many steps to look ahead to determine the label (e.g., 60 for 1 hour).
/// * `threshold`: The percentage change threshold (e.g., 0.005 for 0.5%) to define Buy/Sell labels.
///
/// Returns:
/// * A tuple containing `(features, labels)`, where `features` is a Vec of feature vectors 
///   and `labels` is a Vec of corresponding one-hot encoded labels (`[1.0, 0.0]` for Buy, `[0.0, 1.0]` for Sell).
///   Data points not meeting the threshold are excluded.
pub fn prepare_features_and_labels(
    historical_data: &[MarketData], 
    min_history_points: usize, 
    prediction_horizon: usize, 
    threshold: f64
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    
    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();
    
    if historical_data.len() < min_history_points + prediction_horizon {
        return Err(MLConfigError::InvalidConfig(format!(
            "Insufficient data for preparation: need at least {} points, got {}", 
            min_history_points + prediction_horizon, historical_data.len()
        )).into());
    }

    // Initialize preprocessor
    let mut preprocessor = DataPreprocessor::new(min_history_points)?;
    let mut skipped_warmup = 0;
    let mut skipped_threshold = 0;
    let mut buy_labels = 0;
    let mut sell_labels = 0;

    // Iterate through data to generate features and labels
    for i in 0..historical_data.len() {
        // Convert raw MarketData to TradingMarketData if necessary (assuming direct conversion)
        // If conversion logic is needed, implement it here or adjust input type.
        // For now, assume MarketData has the necessary fields (price, volume, etc.)
        // or implement a From/Into trait.
        // Let's assume we need a conversion/mapping for now:
        let trading_data = TradingMarketData {
             symbol: historical_data[i].symbol.clone(),
             price: historical_data[i].price,
             volume: historical_data[i].volume,
             market_cap: historical_data[i].market_cap,
             price_change_24h: historical_data[i].price_change_24h,
             volume_change_24h: historical_data[i].volume_change_24h,
             timestamp: historical_data[i].timestamp,
             // Assuming the base MarketData doesn't have these, add placeholders or derive if possible
             volume_24h: historical_data[i].volume, // Placeholder
             change_24h: historical_data[i].price_change_24h, // Placeholder
             quote: historical_data[i].quote.clone(), // Assuming quote exists and is cloneable
         };

        // Feed data into preprocessor. We might ignore errors during warmup.
        match preprocessor.process_market_data(&trading_data) {
            Ok(current_features) => {
                // Check if we have enough future data to calculate the label
                if i + prediction_horizon < historical_data.len() {
                    let price_i = historical_data[i].price;
                    let price_future = historical_data[i + prediction_horizon].price;

                    // Avoid division by zero if price_i is zero or very small
                    if price_i.abs() > f64::EPSILON {
                        let change = (price_future - price_i) / price_i;

                        if change > threshold { // Price increased significantly -> Buy label
                            all_features.push(current_features);
                            all_labels.push(vec![1.0, 0.0]);
                            buy_labels += 1;
                        } else if change < -threshold { // Price decreased significantly -> Sell label
                            all_features.push(current_features);
                            all_labels.push(vec![0.0, 1.0]);
                            sell_labels += 1;
                        } else {
                            // Price change was within the threshold -> Skip (Hold)
                            skipped_threshold += 1;
                        }
                    } else {
                         // Skip if current price is zero/too small
                         skipped_threshold += 1;
                    }
                } else {
                    // Not enough future data points left to determine label for this point
                    break; // Stop processing as we can't look far enough ahead
                }
            }
            Err(Error::MLConfigError(MLConfigError::InvalidConfig(msg))) if msg.contains("Insufficient data points") => {
                // Ignore "Insufficient data" errors during the initial warmup phase
                skipped_warmup += 1;
            }
            Err(e) => {
                // Propagate other unexpected errors
                return Err(e);
            }
        }
    }
    
    debug!(
        "Feature/Label Prep: Buy={}, Sell={}, Skipped(Warmup)={}, Skipped(Threshold)={}, TotalInput={}",
        buy_labels, sell_labels, skipped_warmup, skipped_threshold, historical_data.len()
    );

    Ok((all_features, all_labels))
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
        let min_data_points = 26; 
        let mut preprocessor = DataPreprocessor::new(min_data_points)?;
        
        for i in 0..min_data_points {
            let price = 100.0 + (i as f64);
            let volume = 1000.0 + (i as f64 * 10.0);
            let market_cap = 1_000_000_000.0 + (i as f64 * 1_000_000.0);
            let data = create_test_market_data(price, volume, market_cap);
            if i < min_data_points - 1 {
                 let _ = preprocessor.process_market_data(&data);
            } else {
                let features = preprocessor.process_market_data(&data)?;
                assert_eq!(features.len(), 11, "Should have 11 features");
            }
        }
        
        let rsi = preprocessor.calculate_rsi()?;
        assert!(rsi > 50.0, "RSI should indicate upward trend"); 
        
        let last_price = 100.0 + ((min_data_points - 1) as f64);
        let normalized_price = preprocessor.normalize_price(last_price)?;
        assert!((0.0..=1.0).contains(&normalized_price), "Normalized price out of bounds");
        assert!((normalized_price - 1.0).abs() < f64::EPSILON, "Last price should normalize to 1.0");

        Ok(())
    }
} 
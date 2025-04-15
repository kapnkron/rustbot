use crate::utils::error::Result;
use crate::trading::MarketData;
use crate::config::MLConfig;
use tch::{Device, Kind, Tensor, nn};
use std::path::Path;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Prediction {
    pub value: f64,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub features: Vec<f64>,
}

pub struct ModelInference {
    config: MLConfig,
    device: Device,
    var_store: nn::VarStore,
    model: nn::Sequential,
    cache: Arc<Mutex<HashMap<String, Prediction>>>,
    last_prediction_time: Arc<Mutex<DateTime<Utc>>>,
}

impl ModelInference {
    pub async fn new(config: MLConfig) -> Result<Self> {
        let device = Device::cuda_if_available();
        let mut var_store = nn::VarStore::new(device);
        
        let model = nn::seq()
            .add(nn::linear(
                &var_store.root(),
                config.input_size,
                config.hidden_size,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &var_store.root(),
                config.hidden_size,
                config.output_size,
                Default::default(),
            ))
            .add_fn(|xs| xs.softmax(1, Kind::Float));

        Ok(Self {
            config,
            device,
            var_store,
            model,
            cache: Arc::new(Mutex::new(HashMap::new())),
            last_prediction_time: Arc::new(Mutex::new(Utc::now())),
        })
    }

    pub async fn load(&mut self, path: &Path) -> Result<()> {
        self.var_store.load(path)?;
        Ok(())
    }

    pub async fn predict(&mut self, data: &MarketData) -> Result<Prediction> {
        // Check cache first
        let cache_key = format!("{}_{}", data.symbol, data.timestamp.timestamp());
        if let Some(cached) = self.cache.lock().await.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Process features
        let features = self.process_features(data)?;
        
        // Create input tensor
        let input = Tensor::of_slice(&features)
            .reshape(&[1, features.len() as i64])
            .to(self.device);
        
        // Get model output
        let output = self.model.forward(&input);
        let buy_prob = output.double_value(&[0, 0]);
        let sell_prob = output.double_value(&[0, 1]);
        
        // Create prediction
        let prediction = Prediction {
            value: if buy_prob > sell_prob { 1.0 } else { -1.0 },
            confidence: buy_prob.max(sell_prob),
            timestamp: data.timestamp,
            features: features.clone(),
        };

        // Update cache
        self.cache.lock().await.insert(cache_key, prediction.clone());
        
        // Update last prediction time
        *self.last_prediction_time.lock().await = Utc::now();

        Ok(prediction)
    }

    pub async fn predict_batch(&mut self, data: &[MarketData]) -> Result<Vec<Prediction>> {
        let mut predictions = Vec::with_capacity(data.len());
        
        for market_data in data {
            let prediction = self.predict(market_data).await?;
            predictions.push(prediction);
        }
        
        Ok(predictions)
    }

    fn process_features(&self, data: &MarketData) -> Result<Vec<f64>> {
        // Normalize features
        let normalized_price = self.normalize_price(data.price);
        let normalized_volume = self.normalize_volume(data.volume);
        let normalized_market_cap = self.normalize_market_cap(data.market_cap);

        // Combine all features
        Ok(vec![
            normalized_price,
            normalized_volume,
            normalized_market_cap,
            data.price_change_24h,
            data.volume_change_24h,
        ])
    }

    fn normalize_price(&self, price: f64) -> f64 {
        // Simple min-max normalization
        // In a real implementation, you would use proper scaling parameters
        (price - 10000.0) / (100000.0 - 10000.0)
    }

    fn normalize_volume(&self, volume: f64) -> f64 {
        // Simple min-max normalization
        (volume - 1000.0) / (1000000.0 - 1000.0)
    }

    fn normalize_market_cap(&self, market_cap: f64) -> f64 {
        // Simple min-max normalization
        (market_cap - 1_000_000_000.0) / (1_000_000_000_000.0 - 1_000_000_000.0)
    }

    pub async fn clear_cache(&mut self) {
        self.cache.lock().await.clear();
    }

    pub async fn get_last_prediction_time(&self) -> DateTime<Utc> {
        *self.last_prediction_time.lock().await
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

    #[tokio::test]
    async fn test_model_inference() -> Result<()> {
        let config = MLConfig {
            input_size: 5,
            hidden_size: 20,
            output_size: 2,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            confidence_threshold: 0.7,
            training_batch_size: 32,
            training_epochs: 10,
            window_size: 10,
            min_data_points: 100,
            validation_split: 0.2,
            early_stopping_patience: 5,
            save_best_model: true,
        };

        let mut inference = ModelInference::new(config).await?;
        
        // Test single prediction
        let data = create_test_market_data(50000.0, 1000.0, 1_000_000_000.0);
        let prediction = inference.predict(&data).await?;
        
        assert!(prediction.value == 1.0 || prediction.value == -1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert_eq!(prediction.features.len(), 5);
        
        // Test batch prediction
        let batch_data = vec![
            create_test_market_data(50000.0, 1000.0, 1_000_000_000.0),
            create_test_market_data(51000.0, 1100.0, 1_100_000_000.0),
        ];
        
        let predictions = inference.predict_batch(&batch_data).await?;
        assert_eq!(predictions.len(), 2);
        
        Ok(())
    }
} 
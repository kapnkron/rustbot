use crate::error::Result;
use crate::api::MarketData;
use crate::config::MLConfig;
use tch::{Device, Kind, Tensor, nn};
use std::path::Path;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};
use std::collections::HashMap;
use crate::trading::TradingMarketData;
use crate::ml::{ModelConfig, MLConfigError};

#[derive(Debug, Clone)]
pub struct Prediction {
    pub value: f64,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub features: Vec<f64>,
}

pub struct ModelInference {
    config: ModelConfig,
    device: Device,
    var_store: nn::VarStore,
    model: nn::Sequential,
    cache: Arc<Mutex<HashMap<String, Prediction>>>,
    last_prediction_time: Arc<Mutex<DateTime<Utc>>>,
}

impl ModelInference {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let device = crate::ml::get_device();
        let var_store = nn::VarStore::new(device);
        let model = config.architecture.create_model(&var_store.root())?;

        Ok(Self {
            config,
            device,
            var_store,
            model,
            cache: Arc::new(Mutex::new(HashMap::new())),
            last_prediction_time: Arc::new(Mutex::new(Utc::now())),
        })
    }

    pub fn load_model(&mut self, path: &Path) -> Result<()> {
        self.var_store.load(path)?;
        Ok(())
    }

    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if features.is_empty() {
            return Err(MLConfigError::InvalidConfig("Empty features array".to_string()).into());
        }

        let batch_size = features.len() as i64;
        let feature_size = features[0].len() as i64;

        let input = Tensor::f_from_slice(&features.concat())?
            .reshape(&[batch_size, feature_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        
        let predictions = output.to_kind(Kind::Float).to_device(Device::Cpu);
        let mut result = Vec::with_capacity(batch_size as usize);
        
        for i in 0..batch_size {
            let row = predictions.get(i);
            result.push(row.data().to_vec());
        }
        
        Ok(result)
    }

    pub fn predict_single(&self, features: &[f64]) -> Result<Vec<f64>> {
        if features.len() != self.config.architecture.input_size {
            return Err(MLConfigError::InvalidConfig(format!(
                "Feature size mismatch: expected {}, got {}",
                self.config.architecture.input_size,
                features.len()
            )).into());
        }

        let input = Tensor::f_from_slice(features)?
            .reshape(&[1, features.len() as i64])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let prediction = output.to_kind(Kind::Float).to_device(Device::Cpu);
        
        Ok(prediction.data().to_vec())
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

        let mut inference = ModelInference::new(config)?;
        
        // Test single prediction
        let data = create_test_market_data(50000.0, 1000.0, 1_000_000_000.0);
        let prediction = inference.predict(&[data.features()]).await?;
        
        assert!(prediction[0][0] == 1.0 || prediction[0][0] == -1.0);
        assert!(prediction[0][1] >= 0.0 && prediction[0][1] <= 1.0);
        assert_eq!(prediction[0].len(), 5);
        
        // Test batch prediction
        let batch_data = vec![
            create_test_market_data(50000.0, 1000.0, 1_000_000_000.0),
            create_test_market_data(51000.0, 1100.0, 1_100_000_000.0),
        ];
        
        let predictions = inference.predict(&batch_data.iter().map(|d| d.features()).collect::<Vec<_>>()).await?;
        assert_eq!(predictions.len(), 2);
        
        Ok(())
    }
} 
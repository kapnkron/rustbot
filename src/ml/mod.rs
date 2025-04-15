use crate::utils::error::Result;
use tch::{Device, Kind, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::trading::MarketData;
use crate::config::MLConfig;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

mod preprocessing;
pub use preprocessing::DataPreprocessor;

mod evaluation;
pub use evaluation::{ModelEvaluator, ModelMetrics, ConfusionMatrix};

mod versioning;
pub use versioning::{ModelVersion, ModelVersionManager};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: i64,
    pub hidden_size: i64,
    pub output_size: i64,
    pub learning_rate: f64,
    pub model_path: String,
    pub window_size: usize,
}

pub struct TradingModel {
    config: MLConfig,
    model: tch::nn::Sequential,
    device: Device,
    var_store: tch::nn::VarStore,
    preprocessor: DataPreprocessor,
    evaluator: Arc<Mutex<ModelEvaluator>>,
    version_manager: Arc<Mutex<ModelVersionManager>>,
}

impl TradingModel {
    pub fn new(config: MLConfig) -> Result<Self> {
        let device = Device::cuda_if_available();
        let mut var_store = tch::nn::VarStore::new(device);
        
        let model = tch::nn::seq()
            .add(tch::nn::linear(
                &var_store.root(),
                config.input_size,
                config.hidden_size,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(tch::nn::linear(
                &var_store.root(),
                config.hidden_size,
                config.output_size,
                Default::default(),
            ))
            .add_fn(|xs| xs.softmax(1, Kind::Float));

        let evaluator = Arc::new(Mutex::new(ModelEvaluator::new(config.evaluation_window_size)));
        let version_manager = Arc::new(Mutex::new(ModelVersionManager::new(&config.model_path)?));

        Ok(Self {
            config,
            model,
            device,
            var_store,
            preprocessor: DataPreprocessor::new(config.window_size),
            evaluator,
            version_manager,
        })
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        self.var_store.load(path)?;
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.var_store.save(path)?;
        Ok(())
    }

    pub fn process_market_data(&mut self, data: &MarketData) -> Result<Vec<f64>> {
        self.preprocessor.process_market_data(data)
    }

    pub async fn predict(&mut self, data: &MarketData) -> Result<(f64, f64)> {
        let input = self.process_market_data(data)?;
        let start_time = Utc::now();
        
        let output = self.model.forward(&Tensor::of_slice(&input));
        let buy_prob = output.double_value(&[0]);
        let sell_prob = output.double_value(&[1]);
        
        let latency = (Utc::now() - start_time).num_milliseconds() as f64 / 1000.0;
        
        // Record prediction for evaluation
        let mut evaluator = self.evaluator.lock().await;
        evaluator.record_prediction(data.timestamp, buy_prob, sell_prob);
        
        Ok((buy_prob, sell_prob))
    }

    pub fn train_batch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>]) -> Result<f64> {
        let batch_size = features.len() as i64;
        let feature_size = features[0].len() as i64;
        let label_size = labels[0].len() as i64;

        let input = Tensor::of_slice(&features.concat())
            .reshape(&[batch_size, feature_size])
            .to(self.device);
        
        let target = Tensor::of_slice(&labels.concat())
            .reshape(&[batch_size, label_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let loss = output.cross_entropy_for_logits(&target);
        
        loss.backward();
        Ok(loss.double_value(&[]) as f64)
    }

    pub fn train_epoch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>], batch_size: usize) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for i in (0..features.len()).step_by(batch_size) {
            let end = (i + batch_size).min(features.len());
            let batch_features = &features[i..end];
            let batch_labels = &labels[i..end];
            
            let loss = self.train_batch(batch_features, batch_labels)?;
            total_loss += loss;
            count += 1;
        }

        Ok(total_loss / count as f64)
    }

    pub async fn record_actual_move(&mut self, timestamp: DateTime<Utc>, price_change: f64) -> Result<()> {
        let mut evaluator = self.evaluator.lock().await;
        evaluator.record_actual_move(timestamp, price_change);
        evaluator.update_metrics()?;
        Ok(())
    }

    pub async fn get_metrics(&self) -> ModelMetrics {
        let evaluator = self.evaluator.lock().await;
        evaluator.get_metrics().clone()
    }

    pub async fn save_version(&mut self) -> Result<()> {
        let metrics = self.get_metrics().await;
        let mut metrics_map = HashMap::new();
        metrics_map.insert("accuracy".to_string(), metrics.accuracy);
        metrics_map.insert("precision".to_string(), metrics.precision);
        metrics_map.insert("recall".to_string(), metrics.recall);
        metrics_map.insert("f1_score".to_string(), metrics.f1_score);
        metrics_map.insert("roc_auc".to_string(), metrics.roc_auc);

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), self.config.learning_rate.to_string());
        hyperparameters.insert("batch_size".to_string(), self.config.batch_size.to_string());
        hyperparameters.insert("hidden_size".to_string(), self.config.hidden_size.to_string());

        let mut version_manager = self.version_manager.lock().await;
        version_manager.create_version(
            metrics_map,
            &self.config.model_path,
            self.config.training_data_size,
            self.config.features_used.clone(),
            hyperparameters,
        )?;

        version_manager.increment_version();
        Ok(())
    }

    pub async fn get_version_info(&self, version: &str) -> Option<ModelVersion> {
        let version_manager = self.version_manager.lock().await;
        version_manager.get_version(version).cloned()
    }

    pub async fn compare_versions(&self, version1: &str, version2: &str) -> Result<HashMap<String, f64>> {
        let version_manager = self.version_manager.lock().await;
        version_manager.compare_versions(version1, version2)
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
    fn test_model_creation() {
        let config = ModelConfig {
            input_size: 9,
            hidden_size: 20,
            output_size: 2,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            window_size: 10,
        };
        
        assert!(TradingModel::new(config).is_ok());
    }

    #[test]
    fn test_prediction() {
        let config = ModelConfig {
            input_size: 9,
            hidden_size: 20,
            output_size: 2,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            window_size: 10,
        };
        
        let mut model = TradingModel::new(config).unwrap();
        let data = create_test_market_data(100.0, 1000.0, 1_000_000_000.0);
        
        let (buy_prob, sell_prob) = model.predict(&data).unwrap();
        assert!(buy_prob >= 0.0 && buy_prob <= 1.0);
        assert!(sell_prob >= 0.0 && sell_prob <= 1.0);
        assert!((buy_prob + sell_prob - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_model_evaluation_integration() {
        let config = MLConfig {
            input_size: 6,
            hidden_size: 32,
            output_size: 2,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            confidence_threshold: 0.7,
            evaluation_window_size: 100,
            // ... other config fields ...
        };

        let mut model = TradingModel::new(config).unwrap();
        
        // Test prediction and evaluation
        let market_data = MarketData {
            timestamp: Utc::now(),
            symbol: "BTC".to_string(),
            price: 50000.0,
            volume: 1000.0,
            market_cap: 1000000000.0,
            price_change_24h: 0.05,
            volume_change_24h: 0.1,
        };

        let (buy_prob, sell_prob) = model.predict(&market_data).await.unwrap();
        assert!(buy_prob >= 0.0 && buy_prob <= 1.0);
        assert!(sell_prob >= 0.0 && sell_prob <= 1.0);
        
        // Record actual move
        model.record_actual_move(market_data.timestamp, 0.02).await.unwrap();
        
        // Get metrics
        let metrics = model.get_metrics().await;
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert!(metrics.precision >= 0.0 && metrics.precision <= 1.0);
        assert!(metrics.recall >= 0.0 && metrics.recall <= 1.0);
        assert!(metrics.f1_score >= 0.0 && metrics.f1_score <= 1.0);
        assert!(metrics.roc_auc >= 0.0 && metrics.roc_auc <= 1.0);
    }

    #[tokio::test]
    async fn test_model_versioning_integration() -> Result<()> {
        let config = MLConfig {
            input_size: 6,
            hidden_size: 32,
            output_size: 2,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            confidence_threshold: 0.7,
            evaluation_window_size: 100,
            model_path: "models".into(),
            training_data_size: 1000,
            features_used: vec!["price".to_string(), "volume".to_string()],
        };

        let mut model = TradingModel::new(config)?;
        
        // Test prediction and evaluation
        let market_data = MarketData {
            timestamp: Utc::now(),
            symbol: "BTC".to_string(),
            price: 50000.0,
            volume: 1000.0,
            market_cap: 1000000000.0,
            price_change_24h: 0.05,
            volume_change_24h: 0.1,
        };

        let (buy_prob, sell_prob) = model.predict(&market_data).await?;
        model.record_actual_move(market_data.timestamp, 0.02).await?;
        
        // Save version
        model.save_version().await?;
        
        // Get version info
        let version_info = model.get_version_info("1.0.0").await;
        assert!(version_info.is_some());
        let version_info = version_info.unwrap();
        assert_eq!(version_info.version, "1.0.0");
        assert!(version_info.metrics.contains_key("accuracy"));

        Ok(())
    }
} 
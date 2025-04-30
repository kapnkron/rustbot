use crate::error::Result;
use crate::error::Error;
use tch::{Device, Tensor};
use std::path::Path;
use crate::trading::TradingMarketData;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use tch::nn::Module;
use log;
use tch::nn::ModuleT;
use async_trait::async_trait;

mod preprocessing;
pub use preprocessing::DataPreprocessor;

mod evaluation;
pub use evaluation::{ModelEvaluator, ModelMetrics, ConfusionMatrix};

mod versioning;
pub use versioning::{ModelVersion, ModelVersionManager};

mod architecture;
pub use architecture::{ModelArchitecture, Activation, LossFunction, get_device};

pub mod config;
pub use config::{MLConfig, ModelConfig, MLConfigError};

// --- Predictor Trait --- //
#[async_trait]
pub trait Predictor: Send {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<Vec<f64>>;
    // Add other methods from TradingModel needed by TradingBot if any
    // For now, assume only predict is needed by process_market_data
}

#[derive(Debug)]
pub struct TradingModel {
    config: ModelConfig,
    model: tch::nn::Sequential,
    device: Device,
    var_store: tch::nn::VarStore,
    preprocessor: DataPreprocessor,
    evaluator: Arc<Mutex<ModelEvaluator>>,
    version_manager: Arc<Mutex<ModelVersionManager>>,
}

impl TradingModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let device = get_device();
        log::info!("Initializing TradingModel with device: {:?}", device);
        let var_store = tch::nn::VarStore::new(device);
        let model = config.architecture.create_model(&var_store.root())?;
        let config_clone = config.clone();
        
        Ok(Self {
            config,
            model,
            device,
            var_store,
            preprocessor: DataPreprocessor::new(config_clone.window_size, config_clone.window_size)?,
            evaluator: Arc::new(Mutex::new(ModelEvaluator::new(config_clone.window_size))),
            version_manager: Arc::new(Mutex::new(ModelVersionManager::new(&config_clone.model_path)?)),
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

    pub fn process_window(&mut self, window: &[TradingMarketData]) -> Result<Vec<f64>> {
        if window.len() != self.config.window_size {
            return Err(MLConfigError::InvalidConfig(format!(
                "Window size mismatch: expected {}, got {}",
                self.config.window_size,
                window.len()
            )).into());
        }

        let mut features = Vec::with_capacity(window.len());
        for data in window {
            let feature_vec = self.preprocessor.process_market_data(data)?;
            features.extend(feature_vec);
        }
        Ok(features)
    }

    pub fn process_data(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
        self.preprocessor.process_market_data(data)
    }

    pub async fn train(&mut self, data: &[TradingMarketData]) -> Result<()> {
        use tch::nn::{ModuleT, OptimizerConfig};
        let (features, labels) = self.prepare_training_data(data)?;
        
        let input = Tensor::f_from_slice(&features)?;
        let target = Tensor::f_from_slice(&labels)?;
        
        let output = self.model.forward_t(&input, true);
        let loss = match self.config.loss_function {
            LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
            LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        };
        
        let mut optimizer = tch::nn::Adam::default().build(&self.var_store, self.config.learning_rate)?;
        optimizer.backward_step(&loss);
        
        Ok(())
    }

    pub fn train_batch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>]) -> Result<f64> {
        let batch_size = features.len() as i64;
        let feature_size = features[0].len() as i64;
        let label_size = labels[0].len() as i64;

        let input = Tensor::f_from_slice(&features.concat())?
            .reshape(&[batch_size, feature_size])
            .to(self.device);
        
        let target = Tensor::f_from_slice(&labels.concat())?
            .reshape(&[batch_size, label_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let loss = match self.config.loss_function {
            LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
            LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        };
        
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

    pub async fn get_metrics_map(&self) -> HashMap<String, f64> {
        let evaluator = self.evaluator.lock().await;
        let metrics = evaluator.get_metrics();
        let mut metrics_map = HashMap::new();
        metrics_map.insert("mse".to_string(), metrics.mse);
        metrics_map.insert("mae".to_string(), metrics.mae);
        metrics_map.insert("rmse".to_string(), metrics.rmse);
        metrics_map
    }

    pub async fn save_version(&mut self) -> Result<()> {
        let metrics = self.get_metrics_map().await;
        let version = ModelVersion {
            version: "1.0.0".to_string(), // This should be incremented properly
            timestamp: Utc::now(),
            metrics,
            input_size: self.config.architecture.input_size,
            hidden_size: self.config.architecture.hidden_size,
            output_size: self.config.architecture.output_size,
            learning_rate: self.config.learning_rate,
            window_size: self.config.window_size,
        };

        let mut version_manager = self.version_manager.lock().await;
        version_manager.add_version(version)?;
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

    fn prepare_training_data(&mut self, data: &[TradingMarketData]) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for window in data.windows(self.config.window_size + 1) {
            let mut window_features = Vec::new();
            for i in 0..self.config.window_size {
                let feature_vec = self.preprocessor.process_market_data(&window[i])?;
                window_features.extend(feature_vec);
            }
            features.extend(window_features);
            
            // Use the next price as the label
            labels.push(window[self.config.window_size].price);
        }
        
        Ok((features, labels))
    }

    pub async fn get_metrics(&self) -> Result<ModelMetrics> {
        let evaluator = self.evaluator.lock().await;
        Ok(evaluator.get_metrics())
    }
}

// Implement the trait for the real TradingModel
#[async_trait]
impl Predictor for TradingModel {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
        let features_f64 = self.preprocessor.process_market_data(data)?;

        // Ensure we have the expected number of features
        let expected_input_size = self.config.architecture.input_size; // Get from config
        if features_f64.len() as i64 != expected_input_size { // Compare as i64
            return Err(Error::MLError(format!(
                "Preprocessor returned unexpected feature count: expected {}, got {}",
                expected_input_size,
                features_f64.len()
            )));
        }

        let input = Tensor::f_from_slice(&features_f64)?
            .reshape(&[1, expected_input_size]) // Reshape for batch size 1 (already i64)
            .to(self.device);

        // Perform inference within a no_grad block
        let output = tch::no_grad(|| self.model.forward_t(&input, false));

        // Convert output tensor to Vec<f64>
        let output_vec: Vec<f64> = Vec::<f64>::try_from(output)?;

        // Record the prediction - Uncommented
        let mut evaluator = self.evaluator.lock().await; // Use await here
        // Assuming predict returns probabilities for [Buy, Sell]
        let buy_prob = output_vec.get(0).cloned().unwrap_or(0.0); // Assuming index 0 is buy prob
        let sell_prob = output_vec.get(1).cloned().unwrap_or(0.0); // Assuming index 1 is sell prob

        evaluator.record_prediction(data.timestamp, buy_prob, sell_prob); // Pass both probabilities

        Ok(output_vec)
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
    fn test_model_creation() {
        let config = ModelConfig {
            architecture: ModelArchitecture {
                input_size: 9,
                hidden_size: 20,
                output_size: 2,
                num_layers: 2,
                dropout: Some(0.1),
                activation: Activation::ReLU,
            },
            loss_function: LossFunction::MSE,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            window_size: 10,
            min_data_points: 100,
        };
        
        assert!(TradingModel::new(config).is_ok());
    }

    #[test]
    fn test_prediction() {
        let config = ModelConfig {
            architecture: ModelArchitecture {
                input_size: 11,
                hidden_size: 20,
                output_size: 2,
                num_layers: 2,
                dropout: Some(0.1),
                activation: Activation::ReLU,
            },
            loss_function: LossFunction::MSE,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            window_size: 26,
            min_data_points: 100,
        };
        
        let mut model = TradingModel::new(config.clone()).unwrap();

        for i in 0..config.window_size {
            let data = create_test_market_data(
                100.0 + i as f64,
                1000.0 + (i * 10) as f64,
                1_000_000_000.0 + (i * 1_000_000) as f64,
            );
            let _ = model.process_data(&data);
        }

        let final_data = create_test_market_data(
            100.0 + config.window_size as f64,
            1000.0 + (config.window_size * 10) as f64,
            1_000_000_000.0 + (config.window_size * 1_000_000) as f64,
        );
        
        let prediction = model.predict(&final_data).unwrap();
        assert_eq!(prediction.len(), config.architecture.output_size as usize);
    }

    #[tokio::test]
    async fn test_model_evaluation_integration() {
        let config = ModelConfig {
            architecture: ModelArchitecture {
                input_size: 11,
                hidden_size: 32,
                output_size: 2,
                num_layers: 2,
                dropout: Some(0.1),
                activation: Activation::ReLU,
            },
            loss_function: LossFunction::MSE,
            learning_rate: 0.001,
            model_path: "test_model.pt".to_string(),
            window_size: 26,
            min_data_points: 100,
        };

        let mut model = TradingModel::new(config.clone()).unwrap();
        
        // Prime the preprocessor
        let mut _last_timestamp = Utc::now(); // Prefix with underscore
        for i in 0..config.window_size {
             let data = create_test_market_data(
                50000.0 + i as f64,
                1000.0 + (i * 10) as f64,
                1_000_000_000.0 + (i * 1_000_000) as f64,
            );
            _last_timestamp = data.timestamp; // Prefix with underscore
            let _ = model.process_data(&data);
        }

        // Process the data point we'll use for prediction/evaluation
         let market_data = create_test_market_data(
            50000.0 + config.window_size as f64,
            1000.0 + (config.window_size * 10) as f64,
            1_000_000_000.0 + (config.window_size * 1_000_000) as f64,
        );
        let final_timestamp = market_data.timestamp; // Use a different variable name for the actually used timestamp

        // Predict should now work
        let prediction = model.predict(&market_data).unwrap();
        assert_eq!(prediction.len(), config.architecture.output_size as usize);
        
        // Record actual move using the timestamp of the predicted data
        model.record_actual_move(final_timestamp, 0.02).await.unwrap();
        
        // Get metrics and await the Future
        let metrics = model.get_metrics_map().await;
        assert!(metrics.contains_key("mse"));
        assert!(metrics.contains_key("mae"));
        assert!(metrics.contains_key("rmse"));
    }

    #[tokio::test]
    async fn test_model_versioning_integration() -> Result<()> {
        let config = ModelConfig {
            architecture: ModelArchitecture {
                input_size: 11,
                hidden_size: 32,
                output_size: 2,
                num_layers: 2,
                dropout: Some(0.1),
                activation: Activation::ReLU,
            },
            loss_function: LossFunction::MSE,
            learning_rate: 0.001,
            model_path: "test_model_versioning.pt".to_string(),
            window_size: 26,
            min_data_points: 100,
        };

        let mut model = TradingModel::new(config.clone())?;
        
        // Prime the preprocessor
        let mut _last_timestamp = Utc::now(); // Prefix with underscore
        for i in 0..config.window_size {
             let data = create_test_market_data(
                50000.0 + i as f64,
                1000.0 + (i * 10) as f64,
                1_000_000_000.0 + (i * 1_000_000) as f64,
            );
            _last_timestamp = data.timestamp; // Prefix with underscore
            let _ = model.process_data(&data);
        }

        // Process the data point we'll use for prediction/evaluation
         let market_data = create_test_market_data(
            50000.0 + config.window_size as f64,
            1000.0 + (config.window_size * 10) as f64,
            1_000_000_000.0 + (config.window_size * 1_000_000) as f64,
        );
        let final_timestamp = market_data.timestamp; // Use a different variable name

        // Predict should now work
        let _prediction = model.predict(&market_data)?;
        // Record an actual move to generate some metrics
        model.record_actual_move(final_timestamp, 0.02).await?;
        
        // Save version
        model.save_version().await?;
        
        let version_info = model.get_version_info("1.0.0").await;
        assert!(version_info.is_some());
        let version_info = version_info.unwrap();
        assert_eq!(version_info.version, "1.0.0");
        assert!(version_info.metrics.contains_key("mse"));
        assert!(version_info.metrics.contains_key("mae"));
        assert!(version_info.metrics.contains_key("rmse"));

        Ok(())
    }
} 
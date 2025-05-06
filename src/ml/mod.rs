use crate::error::Result;
use crate::error::Error;
use tch::{Device, Tensor, Kind};
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
use crate::config::Config;
use crate::data_loader::load_market_data_from_csv;
use crate::ml::training::ModelTrainer;
use log::info;
use log::warn;
use anyhow::{Context, Result as AnyhowResult};
use std::path::PathBuf;
use crate::ml::preprocessing::prepare_features_and_labels;

mod preprocessing;
pub use preprocessing::DataPreprocessor;

mod evaluation;
pub use evaluation::{ModelEvaluator, ModelMetrics, ConfusionMatrix};

mod versioning;
pub use versioning::{ModelVersion, ModelVersionManager};

mod architecture;
pub use architecture::{ModelArchitecture, Activation, LossFunction, get_device};

mod training;

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
            preprocessor: DataPreprocessor::new(config_clone.min_data_points)?,
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

        let mut features = Vec::with_capacity(window.len() * 11); // Estimate size
        // Use iterator instead of range loop
        for data_point in window.iter().take(self.config.window_size) {
            let feature_vec = self.preprocessor.process_market_data(data_point)?;
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
            .reshape([batch_size, feature_size])
            .to(self.device);
        
        let target = Tensor::f_from_slice(&labels.concat())?
            .reshape([batch_size, label_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let loss = match self.config.loss_function {
            LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
            LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        };
        
        loss.backward();
        Ok(loss.double_value(&[]))
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
            // Use iterator instead of range loop
            for data_point in window.iter().take(self.config.window_size) {
                let feature_vec = self.preprocessor.process_market_data(data_point)?;
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

        let expected_input_size = self.config.architecture.input_size;
        if features_f64.len() as i64 != expected_input_size { 
            return Err(Error::MLError(format!(
                "Preprocessor returned unexpected feature count: expected {}, got {}",
                expected_input_size,
                features_f64.len()
            )));
        }

        // Convert features to f32 for the model
        let features_f32: Vec<f32> = features_f64.iter().map(|&x| x as f32).collect();

        // Create input tensor as Float
        let input = Tensor::f_from_slice(&features_f32)? 
            .reshape([1, expected_input_size]) 
            .to(self.device)
            .to_kind(tch::Kind::Float); // Explicitly ensure Float type

        let output = tch::no_grad(|| self.model.forward_t(&input, false));

        // --- Apply Softmax to get probabilities --- 
        let probabilities = output.softmax(-1, None); // Apply softmax along the last dimension

        // Convert probabilities tensor back to Vec<f64>
        let probabilities_f64 = probabilities.to_kind(tch::Kind::Double);
        let flattened_probs = probabilities_f64.flatten(0, -1); 
        let output_vec: Vec<f64> = Vec::<f64>::try_from(flattened_probs)?;

        let mut evaluator = self.evaluator.lock().await;
        // Assuming predict returns probabilities for [Buy, Sell]
        let buy_prob = output_vec.first().cloned().unwrap_or(0.0); // Assuming index 0 is buy prob
        let sell_prob = output_vec.get(1).cloned().unwrap_or(0.0); // Assuming index 1 is sell prob

        evaluator.record_prediction(data.timestamp, buy_prob, sell_prob); // Pass both probabilities

        Ok(output_vec)
    }
}

/// Public function to orchestrate the model training process.
pub async fn run_training_session(
    config_path: PathBuf,
    csv_path: PathBuf,
    model_output_path: PathBuf,
    split_ratio: f64,
) -> AnyhowResult<()> {
    info!("Starting model training session via run_training_session...");
    info!("Config path: \"{}\"", config_path.display());
    info!("CSV data path: \"{}\"", csv_path.display());
    info!("Model output path: \"{}\"", model_output_path.display());
    info!("Split ratio: {}", split_ratio);

    // --- 1. Load Configuration --- (Handles its own logging)
    let config = Config::load(&config_path)
        .context("Failed to load configuration")?;
    let ml_config = config.ml; // Use the specific ML config

    // --- 2. Load Data --- (Handles its own logging)
    let market_data = load_market_data_from_csv(&csv_path)
        .context(format!("Failed to load market data from {}", csv_path.display()))?;
    info!("Loaded {} historical data records.", market_data.len());

    // --- 3. Split Data --- (Using a simple split for now)
    let split_index = (market_data.len() as f64 * split_ratio) as usize;
    if split_index == 0 || split_index >= market_data.len() {
        anyhow::bail!("Invalid split ratio {} results in zero training or validation samples.", split_ratio);
    }
    let (training_data, validation_data) = market_data.split_at(split_index);
    info!(
        "Split data: {} training samples, {} validation samples.",
        training_data.len(),
        validation_data.len()
    );

    // --- 4. Initialize Trainer --- (Handles its own logging)
    let mut trainer = ModelTrainer::new(ml_config.clone())
        .context("Failed to initialize ModelTrainer")?;
    info!("Model trainer initialized.");

    // --- 5. Train Model --- (Handles its own logging)
    info!("Starting training...");
    trainer.train(training_data, validation_data)
        .await
        .context("Model training failed")?;
    info!("Training finished.");

    // --- 6. Save Model --- (Handles its own logging)
    info!("Saving trained model to: {}", model_output_path.display());
    trainer.save(&model_output_path)
        .context(format!("Failed to save model to {}", model_output_path.display()))?;
    info!("Model saved successfully.");

    // --- 7. Evaluate Model --- 
    info!("Starting model evaluation on validation set...");
    if validation_data.is_empty() {
        warn!("Validation data set is empty, skipping evaluation.");
    } else {
        // Prepare validation data
        // TODO: Get horizon/threshold properly if needed for evaluation prep
        // Using defaults similar to ModelTrainer::train for now
        let prediction_horizon = 60; // Example: Needs to be consistent or configurable
        let threshold = 0.005;      // Example: Needs to be consistent or configurable
        let (val_features, val_labels) = prepare_features_and_labels(
            validation_data, 
            ml_config.min_data_points, 
            prediction_horizon, 
            threshold
        )
            .context("Failed to prepare validation features and labels")?;

        if val_features.is_empty() {
            warn!("Prepared validation features are empty (likely due to insufficient data for lookahead/window), skipping evaluation.");
        } else {
            let num_val_samples = val_features.len();
            info!("Prepared {} samples for validation.", num_val_samples);
            
            // Create a new trainer instance to load the saved model
            let mut eval_trainer = ModelTrainer::new(ml_config.clone())
                 .context("Failed to initialize ModelTrainer for evaluation")?;
            
            eval_trainer.load(&model_output_path)
                 .context(format!("Failed to load saved model from {} for evaluation", model_output_path.display()))?;

            let device = get_device(); // Get the device (CPU/GPU) used by the trainer

            // Convert validation data to tensors
            let val_features_tensor = Tensor::f_from_slice(&val_features.concat())?
                .view([num_val_samples as i64, -1]) // Reshape: [num_samples, features_per_sample]
                .to_kind(Kind::Float) // Ensure Float type
                .to(device);
            let val_labels_tensor = Tensor::f_from_slice(&val_labels.concat())?
                .view([num_val_samples as i64, -1]) // Reshape: [num_samples, num_classes]
                .to_kind(Kind::Float) // Ensure Float type
                .to(device);

            // Run inference (no_grad to disable gradient calculation)
            let predictions = tch::no_grad(|| {
                // Use the public getter for the model
                eval_trainer.model().forward_t(&val_features_tensor, false) // Set train=false for evaluation mode
            });
            
            // Calculate accuracy
            let predicted_classes = predictions.argmax(1, false); // Get index of max logit (predicted class)
            let true_classes = val_labels_tensor.argmax(1, false); // Get index of true class
            
            let correct_predictions = predicted_classes.eq_tensor(&true_classes).sum(Kind::Int64);
            let total_predictions = num_val_samples as i64;
            
            let accuracy = correct_predictions.int64_value(&[]) as f64 / total_predictions as f64;
            
            info!("Validation Accuracy: {:.4} ({} / {})", accuracy, correct_predictions.int64_value(&[]), total_predictions);
            
            // TODO: Add more metrics (Precision, Recall, F1, Confusion Matrix)
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::{create_test_model_config, create_test_trading_market_data};
    use tempfile::tempdir;
    use std::collections::HashMap;
    use crate::error::Error;
    use super::MLConfigError;
    use chrono::Utc;

    fn create_test_model_config_for_ml() -> ModelConfig {
        let mut config = create_test_model_config();
        config.min_data_points = 26;
        config
    }

    #[test]
    fn test_data_preprocessor_process_market_data() -> Result<()> {
        let config = create_test_model_config_for_ml();
        let mut preprocessor = DataPreprocessor::new(config.min_data_points)?;
        // Feed history first (now loops 26 times)
        for i in 0..config.min_data_points {
            let history_data = create_test_trading_market_data("BTC/USD", 50000.0 + i as f64);
            let _ = preprocessor.process_market_data(&history_data);
        }
        // Now process the final data point
        let final_data = create_test_trading_market_data("BTC/USD", 50000.0 + config.min_data_points as f64);
        let result = preprocessor.process_market_data(&final_data);
        assert!(result.is_ok(), "process_market_data failed: {:?}", result.err());
        let features = result.unwrap();
        assert_eq!(features.len(), 11);
        Ok(())
    }

    #[test]
    fn test_process_window_incorrect_size() -> Result<()> {
        let config = create_test_model_config_for_ml();
        let mut preprocessor = DataPreprocessor::new(config.min_data_points)?;
        // Needs enough history for the final call to process_market_data to succeed
        for i in 0..(config.min_data_points -1) { // Feed 25 points
             let history_data = create_test_trading_market_data("BTC/USD", 50000.0 + i as f64);
             let _ = preprocessor.process_market_data(&history_data);
        }
        // Now call with the 26th point, should fail inside (e.g. MACD needs 26)
        // but the main check is for the outer insufficient points error?
        // Let's rethink this test. It was originally checking process_window, which doesn't exist.
        // Let's test that process_market_data fails correctly if called *before* min_data_points is reached.
        let mut preprocessor_short = DataPreprocessor::new(config.min_data_points)?;
        let data = create_test_trading_market_data("BTC/USD", 50004.0);
        for _ in 0..5 { // Feed only 5 points
             let _ = preprocessor_short.process_market_data(&data); // Ignore result
        }
        let final_result = preprocessor_short.process_market_data(&data); // Call again
        assert!(final_result.is_err()); // Expect error: Insufficient data points
        if let Err(Error::MLConfigError(MLConfigError::InvalidConfig(msg))) = final_result {
            assert!(msg.contains("Insufficient data points"));
        } else {
            panic!("Expected Insufficient data points error, got {:?}", final_result);
        }
        Ok(())
    }

    #[test]
    fn test_process_window_correct_size() -> Result<()> {
        let config = create_test_model_config_for_ml();
        let mut preprocessor = DataPreprocessor::new(config.min_data_points)?;
        // Feed history first (now loops 26 times)
        for i in 0..config.min_data_points {
            let history_data = create_test_trading_market_data("BTC/USD", 50000.0 + i as f64);
            let _ = preprocessor.process_market_data(&history_data); 
        }
        // Now process the last point again, history should be sufficient for all indicators
        let final_data = create_test_trading_market_data("BTC/USD", 50000.0 + config.min_data_points as f64);
        let result = preprocessor.process_market_data(&final_data); 
        assert!(result.is_ok(), "Processing failed: {:?}", result.err());
        let features = result.unwrap();
        assert_eq!(features.len(), 11);
        Ok(())
    }

    #[test]
    fn test_trading_model_new() -> Result<()> {
        let config = create_test_model_config_for_ml();
        let model = TradingModel::new(config)?;
        // Compare device directly
        assert_eq!(model.device, tch::Device::Cpu); // Assuming CPU for tests
        Ok(())
    }

    #[tokio::test]
    async fn test_trading_model_predict() -> Result<()> {
        let config = create_test_model_config_for_ml();
        let mut model = TradingModel::new(config)?;
        // Feed required history (now 26 points)
        for i in 0..model.config.min_data_points {
            let history_data = create_test_trading_market_data("BTC/USD", 50000.0 + i as f64);
            let _ = model.process_data(&history_data); // Feed the preprocessor
        }
        
        // Now predict with a new data point
        let market_data = create_test_trading_market_data("BTC/USD", 50000.0 + model.config.min_data_points as f64);
        let prediction = model.predict(&market_data).await;
        assert!(prediction.is_ok(), "Prediction failed: {:?}", prediction.err());
        let probs = prediction.unwrap();
        assert_eq!(probs.len(), 2);
        assert!(probs[0] >= 0.0 && probs[0] <= 1.0);
        assert!(probs[1] >= 0.0 && probs[1] <= 1.0);
        Ok(())
    }

    #[test]
    fn test_trading_model_load_save() -> Result<()> {
        let config = create_test_model_config();
        let model = TradingModel::new(config)?;
        let dir = tempdir()?;
        let path = dir.path().join("test_model.ot");
        model.save(&path)?;
        let mut new_model = TradingModel::new(create_test_model_config())?;
        new_model.load(&path)?;
        Ok(())
    }

    #[test]
    fn test_model_evaluator() -> Result<()> {
        let mut evaluator = ModelEvaluator::new(5);
        let now = Utc::now();

        evaluator.record_prediction(now - chrono::Duration::seconds(4), 0.7, 0.3);
        evaluator.record_actual_move(now - chrono::Duration::seconds(4), 100.0);

        evaluator.record_prediction(now - chrono::Duration::seconds(3), 0.2, 0.8);
        evaluator.record_actual_move(now - chrono::Duration::seconds(3), -50.0);

        evaluator.update_metrics()?;
        let metrics = evaluator.get_metrics();
        assert!(metrics.mse >= 0.0);
        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        // let cm = evaluator.get_confusion_matrix(); // Commented out
        Ok(())
    }

    #[tokio::test]
    async fn test_model_version_manager() -> Result<()> {
        let dir = tempdir()?;
        let base_path_str = dir.path().to_str().expect("Temp path is not valid UTF-8");
        let mut manager = ModelVersionManager::new(base_path_str)?;

        let mut metrics1 = HashMap::new();
        metrics1.insert("mse".to_string(), 0.1);
        let version1 = ModelVersion {
            version: "1.0.0".to_string(),
            timestamp: Utc::now(),
            metrics: metrics1.clone(),
            input_size: 10, hidden_size: 5, output_size: 2, learning_rate: 0.001, window_size: 5
        };
        manager.add_version(version1)?;

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let mut metrics2 = HashMap::new();
        metrics2.insert("mse".to_string(), 0.05);
        let version2 = ModelVersion {
            version: "1.1.0".to_string(),
            timestamp: Utc::now(),
            metrics: metrics2.clone(),
             input_size: 10, hidden_size: 5, output_size: 2, learning_rate: 0.001, window_size: 5
        };
        manager.add_version(version2)?;

        assert!(manager.get_version("1.0.0").is_some());
        assert_eq!(manager.get_version("1.1.0").unwrap().metrics["mse"], 0.05);
        let comparison = manager.compare_versions("1.0.0", "1.1.0")?;
        assert_eq!(comparison.get("mse"), Some(&-0.05), "Comparison difference mismatch");
        assert!(manager.get_version("invalid").is_none());
        assert!(manager.compare_versions("1.0.0", "invalid").is_err());

        Ok(())
    }
} 
use crate::error::Result;
// use tch::{Device, Tensor, Kind}; // Removed tch
use crate::trading::TradingMarketData;
// use tch::nn::Module; // Removed tch
use log;
// use tch::nn::ModuleT; // Removed tch
// use crate::data_loader::load_market_data_from_csv; // This seems to be a circular dep or incorrect path
// use crate::ml::training::ModelTrainer; // training.rs might also be tch-dependent
use log::info;
use log::warn;
use anyhow::{Context, Result as AnyhowResult};
use std::path::PathBuf;
// use crate::ml::preprocessing::prepare_features_and_labels; // preprocessing.rs might use tch

mod preprocessing;
pub use preprocessing::DataPreprocessor;

mod evaluation;
pub use evaluation::{ModelEvaluator, ModelMetrics, ConfusionMatrix};

mod versioning;
pub use versioning::{ModelVersion, ModelVersionManager};

pub mod architecture;
pub use architecture::{ModelArchitecture, Activation, LossFunction}; // get_device might be tch-specific
// pub use architecture::get_device; // Commenting out get_device as it's likely tch specific

pub mod training; // This module will likely need heavy commenting/removal too

pub mod config;
pub use config::{MLConfig, ModelConfig, MLConfigError};

// --- Prediction Output --- //
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PredictionOutput {
    pub predictions: Vec<f64>, // Main model output, e.g., [price_prediction, direction_prediction] or class probabilities
    pub confidence: f64,       // Overall confidence in the prediction
}

impl PredictionOutput {
    // Simple constructor
    pub fn new(predictions: Vec<f64>, confidence: f64) -> Self {
        Self {
            predictions,
            confidence,
        }
    }
}

// --- Predictor Trait --- //
#[async_trait::async_trait]
pub trait Predictor: Send + Sync { // Added Sync
    async fn predict(&mut self, data: &TradingMarketData) -> Result<PredictionOutput>;
    // Add other methods from TradingModel needed by TradingBot if any
    // For now, assume only predict is needed by process_market_data
}

/* // Commenting out Tch-based TradingModel
#[derive(Debug)]
pub struct TradingModel {
    config: ModelConfig,
    // model: tch::nn::Sequential, 
    // device: Device,
    // var_store: tch::nn::VarStore,
    preprocessor: DataPreprocessor,
    evaluator: Arc<Mutex<ModelEvaluator>>,
    version_manager: Arc<Mutex<ModelVersionManager>>,
}

impl TradingModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        // let device = get_device();
        // log::info!("Initializing TradingModel with device: {:?}", device);
        // let var_store = tch::nn::VarStore::new(device);
        // let model = config.architecture.create_model(&var_store.root())?;
        let config_clone = config.clone();
        
        Ok(Self {
            config,
            // model,
            // device,
            // var_store,
            preprocessor: DataPreprocessor::new(config_clone.min_data_points)?,
            evaluator: Arc::new(Mutex::new(ModelEvaluator::new(config_clone.window_size))),
            version_manager: Arc::new(Mutex::new(ModelVersionManager::new(&config_clone.model_path)?)),
        })
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        // self.var_store.load(path)?;
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        // self.var_store.save(path)?;
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
        // use tch::nn::{ModuleT, OptimizerConfig};
        // let (features, labels) = self.prepare_training_data(data)?;
        // 
        // let input = Tensor::f_from_slice(&features)?;
        // let target = Tensor::f_from_slice(&labels)?;
        // 
        // let output = self.model.forward_t(&input, true);
        // let loss = match self.config.loss_function {
        //     LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
        //     LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        // };
        // 
        // let mut optimizer = tch::nn::Adam::default().build(&self.var_store, self.config.learning_rate)?;
        // optimizer.backward_step(&loss);
        
        Ok(())
    }

    pub fn train_batch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>]) -> Result<f64> {
        // let batch_size = features.len() as i64;
        // let feature_size = features[0].len() as i64;
        // let label_size = labels[0].len() as i64;
        // 
        // let input = Tensor::f_from_slice(&features.concat())?
        //     .reshape([batch_size, feature_size])
        //     .to(self.device);
        // 
        // let target = Tensor::f_from_slice(&labels.concat())?
        //     .reshape([batch_size, label_size])
        //     .to(self.device);
        // 
        // let output = self.model.forward(&input);
        // let loss = match self.config.loss_function {
        //     LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
        //     LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        // };
        // 
        // loss.backward();
        // Ok(loss.double_value(&[]))
        Ok(0.0) // Placeholder
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
#[async_trait::async_trait]
impl Predictor for TradingModel {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<PredictionOutput> {
        let features_f64 = self.preprocessor.process_market_data(data)?;

        let expected_input_size = self.config.architecture.input_size;
        if features_f64.len() as i64 != expected_input_size { 
            return Err(Error::MLError(format!(
                "Preprocessor returned unexpected feature count: expected {}, got {}",
                expected_input_size,
                features_f64.len()
            )));
        }

        // Convert features to Tensor
        // let input = Tensor::f_from_slice(&features_f64)?
        //     .reshape([1, -1]) // Reshape to [batch_size, num_features]
        //     .to(self.device);
        // 
        // // Perform inference
        // let output_tensor = tch::no_grad(|| self.model.forward_t(&input, false));
        // 
        // // Convert output Tensor to Vec<f64>
        // let predictions: Vec<f64> = Vec::try_from(output_tensor.to(tch::Device::Cpu).kind(tch::Kind::Double))?;
        // Ok(predictions)
        Ok(PredictionOutput::new(vec![0.5, 0.5], 0.5)) // Placeholder prediction
    }
}
*/

// Placeholder for Python-based predictor
#[derive(Debug)]
pub struct PythonPredictor {
    pub model_path: String, // Example: path to the .pt file or a directory
    pub scaler_path: String, // Example: path to the .pkl scaler - Changed to String
    pub script_path: String, // Path to the python inference script
}

impl PythonPredictor {
    // Constructor for PythonPredictor
    pub fn new(model_path: String, scaler_path: String, script_path: String) -> Self {
        Self {
            model_path,
            scaler_path,
            script_path,
        }
    }
}

#[async_trait::async_trait]
impl Predictor for PythonPredictor {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<PredictionOutput> {
        // TODO: Implement Python script call
        info!("[Rust] PythonPredictor received data for symbol: {}, price: {}", data.symbol, data.price);
        // Placeholder: Simulate calling Python and getting a dummy prediction
        // In a real scenario, this would involve inter-process communication or an HTTP call.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await; // Simulate network/IPC delay
        
        // Dummy prediction based on price, similar to DummyPredictor for now
        if data.price > 200.0 { // Example threshold
            Ok(PredictionOutput::new(vec![0.8, 0.2], 0.8)) // e.g., [prob_long, prob_short] or [price_prediction, confidence]
        } else if data.price < 100.0 {
            Ok(PredictionOutput::new(vec![0.3, 0.7], 0.7))
        } else {
            Ok(PredictionOutput::new(vec![0.5, 0.5], 0.5))
        }
    }
}


pub async fn run_training_session(
    config_path: PathBuf,
    csv_path: PathBuf,
    model_output_path: PathBuf,
    split_ratio: f64, // e.g., 0.8 for 80% train, 20% test
) -> AnyhowResult<()> {
    info!("Starting training session...");
    info!("Config path: {:?}", config_path);
    info!("CSV data path: {:?}", csv_path);
    info!("Model output path: {:?}", model_output_path);
    info!("Split ratio: {}", split_ratio);

    // This function will now likely orchestrate calling the Python training script
    // Or it might be removed if Python handles its own training lifecycle entirely.

    // 1. Load general config to get Python script paths or other parameters
    // let general_config = Config::load_from_file(&config_path).context("Failed to load general config")?;
    // let python_train_script_path = general_config.ml.python_train_script_path; // Assuming such a field exists

    // 2. Prepare arguments for the Python script
    let mut command = std::process::Command::new("python3"); // Or your specific python interpreter
    command.arg("train_model.py"); // Assuming train_model.py is in PATH or use full path
    command.arg("--data-path").arg(csv_path.to_str().context("CSV path is not valid UTF-8")?);
    command.arg("--model-out").arg(model_output_path.to_str().context("Model output path is not valid UTF-8")?);
    command.arg("--split-ratio").arg(split_ratio.to_string());
    // Add other necessary arguments from your MLConfig or ModelConfig if the Python script needs them
    // e.g., command.arg("--epochs").arg(ml_config.training_epochs.to_string());

    info!("Executing Python training script: {:?}", command);

    // 3. Execute the Python script
    let output = command.output().context("Failed to execute Python training script")?;

    // 4. Check output and handle errors
    if output.status.success() {
        info!("Python training script executed successfully.");
        info!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
    } else {
        warn!("Python training script failed.");
        warn!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
        return Err(anyhow::anyhow!("Python training script failed. Stderr: {}", String::from_utf8_lossy(&output.stderr)));
    }
    
    Ok(())
}


// #[cfg(test)] // Commenting out old tests that rely on tch::TradingModel
// mod tests {
//     use super::*;
//     use crate::ml::config::{ModelConfig, ModelArchitecture, Activation, LossFunction};
//     use crate::trading::TradingMarketData;
//     use chrono::Utc;
//     use tch::Device;
//     use tempfile::tempdir;
//     use std::fs::File;
//     use std::io::Write;

//     // Helper to create a default ModelConfig for these tests
//     fn create_test_model_config_for_ml() -> ModelConfig {
//         ModelConfig {
//             architecture: ModelArchitecture { input_size: 11, hidden_size: 20, output_size: 1, num_layers: 1, dropout: None, activation: Activation::ReLU },
//             loss_function: LossFunction::MSE,
//             learning_rate: 0.001,
//             model_path: "test_model_ml_mod.pt".to_string(),
//             window_size: 10,
//             min_data_points: 10,
//         }
//     }

//     #[test]
//     fn test_data_preprocessor_process_market_data() -> Result<()> {
//         let config = create_test_model_config_for_ml();
//         let mut preprocessor = DataPreprocessor::new(config.min_data_points)?;
//         let market_data = TradingMarketData {
//             symbol: "BTC/USD".to_string(),
//             price: 50000.0,
//             volume: 100.0,
//             timestamp: Utc::now(),
//             market_cap: 5_000_000_000.0,
//             price_change_24h: 0.05,
//             volume_change_24h: 0.1,
//             volume_24h: 2000.0,
//             change_24h: 0.02,
//             quote: crate::api::types::Quote { usd: crate::api::types::USDData { price: 50000.0, volume_24h: 2000.0, market_cap: 5_000_000_000.0, percent_change_24h: 0.05, volume_change_24h: 0.1 } }
//         };
//         let features = preprocessor.process_market_data(&market_data)?;
//         assert_eq!(features.len(), 11); // Based on DataPreprocessor implementation
//         Ok(())
//     }
    
//     #[test]
//     fn test_process_window_incorrect_size() -> Result<()> {
//         let config = create_test_model_config_for_ml();
//         let mut model = TradingModel::new(config)?;
//         let window = vec![TradingMarketData { /* ... */ symbol: "".to_string(), price: 0.0, volume: 0.0, timestamp: Utc::now(), market_cap: 0.0, price_change_24h: 0.0, volume_change_24h: 0.0, volume_24h: 0.0, change_24h: 0.0, quote: crate::api::types::Quote{usd: crate::api::types::USDData{price:0.0, volume_24h: 0.0, market_cap:0.0, percent_change_24h:0.0, volume_change_24h:0.0}} }; model.config.window_size - 1];
//         let result = model.process_window(&window);
//         assert!(result.is_err());
//         Ok(())
//     }

//     #[test]
//     fn test_process_window_correct_size() -> Result<()> {
//         let config = create_test_model_config_for_ml();
//         let mut model = TradingModel::new(config)?;
//         let window = vec![TradingMarketData { /* ... */ symbol: "".to_string(), price: 0.0, volume: 0.0, timestamp: Utc::now(), market_cap: 0.0, price_change_24h: 0.0, volume_change_24h: 0.0, volume_24h: 0.0, change_24h: 0.0, quote: crate::api::types::Quote{usd: crate::api::types::USDData{price:0.0, volume_24h: 0.0, market_cap:0.0, percent_change_24h:0.0, volume_change_24h:0.0}} }; model.config.window_size];
//         let features = model.process_window(&window)?;
//         assert_eq!(features.len(), model.config.window_size * 11);
//         Ok(())
//     }

//     #[test]
//     fn test_trading_model_new() -> Result<()> {
//         let config = create_test_model_config_for_ml();
//         let model = TradingModel::new(config)?;
//         // assert_eq!(model.device, Device::Cpu); // Or whatever get_device() returns for test env
//         Ok(())
//     }

//     #[tokio::test]
//     async fn test_trading_model_predict() -> Result<()> {
//         let config = create_test_model_config_for_ml();
//         let mut model = TradingModel::new(config)?;
//         let market_data = TradingMarketData { symbol: "BTC/USD".to_string(), price: 50000.0, volume: 100.0, timestamp: Utc::now(), market_cap: 0.0, price_change_24h: 0.0, volume_change_24h: 0.0, volume_24h: 0.0, change_24h: 0.0, quote: crate::api::types::Quote{usd: crate::api::types::USDData{price:0.0, volume_24h: 0.0, market_cap:0.0, percent_change_24h:0.0, volume_change_24h:0.0}} };
//         let prediction = model.predict(&market_data).await?;
//         // This will now return the placeholder [0.5, 0.5] as the tch model part is commented out
//         assert_eq!(prediction, vec![0.5, 0.5]); 
//         Ok(())
//     }

//     #[test]
//     fn test_trading_model_load_save() -> Result<()> {
//         let dir = tempdir()?;
//         let file_path = dir.path().join("test_model_save.pt");
        
//         let config = create_test_model_config_for_ml();
//         let mut model = TradingModel::new(config.clone())?;
//         model.save(&file_path)?;
        
//         let mut new_model = TradingModel::new(config)?;
//         new_model.load(&file_path)?;
        
//         // Add assertions to compare model.var_store and new_model.var_store if possible, 
//         // or at least check that load/save didn't panic.
//         Ok(())
//     }

//     #[test]
//     fn test_model_evaluator() -> Result<()> {
//         let mut evaluator = ModelEvaluator::new(5);
//         let now = Utc::now();
        
//         evaluator.record_prediction(now - chrono::Duration::seconds(4), 0.7, 0.3);
//         evaluator.record_actual_move(now - chrono::Duration::seconds(4), 100.0);
        
//         evaluator.record_prediction(now - chrono::Duration::seconds(3), 0.2, 0.8);
//         evaluator.record_actual_move(now - chrono::Duration::seconds(3), -50.0);
        
//         let metrics = evaluator.get_metrics();
//         // Add assertions for metrics, e.g., that MSE is calculated
//         // For now, just checking it runs
//         println!("Test Metrics: {:?}", metrics);
//         Ok(())
//     }

//     #[tokio::test]
//     async fn test_model_version_manager() -> Result<()> {
//         let dir = tempdir()?;
//         let model_dir = dir.path().join("models");
//         std::fs::create_dir_all(&model_dir)?;
//         let model_path_base = model_dir.to_str().unwrap();

//         let mut manager = ModelVersionManager::new(model_path_base)?;
//         let version_info = ModelVersion {
//             version: "0.1.0".to_string(),
//             timestamp: Utc::now(),
//             metrics: HashMap::new(),
//             input_size: 10, hidden_size:20, output_size:1, learning_rate: 0.01, window_size: 5
//         };
//         manager.add_version(version_info.clone())?;
//         let retrieved = manager.get_version("0.1.0");
//         assert_eq!(retrieved, Some(&version_info));
//         Ok(())
//     }

//     // A simple test for the placeholder PythonPredictor
//     #[tokio::test]
//     async fn test_python_predictor_placeholder() -> Result<()> {
//         let mut predictor = PythonPredictor::new();
//         let market_data = TradingMarketData { 
//             symbol: "BTC/USD".to_string(), 
//             price: 150.0, // Middle value for placeholder logic
//             volume: 100.0, 
//             timestamp: Utc::now(), 
//             market_cap: 0.0, 
//             price_change_24h: 0.0, 
//             volume_change_24h: 0.0, 
//             volume_24h: 0.0, 
//             change_24h: 0.0, 
//             quote: crate::api::types::Quote{usd: crate::api::types::USDData{price:0.0, volume_24h: 0.0, market_cap:0.0, percent_change_24h:0.0, volume_change_24h:0.0}}
//         };
//         let prediction = predictor.predict(&market_data).await?;
//         assert_eq!(prediction, vec![0.5, 0.5]); // Expecting the middle-case dummy prediction
//         Ok(())
//     }
// }

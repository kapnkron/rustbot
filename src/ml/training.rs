use crate::error::Result;
use crate::api::MarketData;
use crate::config::MLConfig;
use tch::{Device, Kind, Tensor, nn};
use std::path::Path;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};
use super::architecture::{ModelArchitecture, Activation, LossFunction, get_device};
use crate::trading::TradingMarketData;
use crate::ml::{ModelConfig, MLConfigError};
use crate::ml::preprocessing::prepare_features_and_labels;

pub struct ModelTrainer {
    config: ModelConfig,
    device: Device,
    var_store: nn::VarStore,
    model: nn::Sequential,
    optimizer: nn::Optimizer,
    best_validation_loss: f64,
    patience_counter: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub loss_function: LossFunction,
    pub learning_rate: f64,
    pub model_path: String,
    pub window_size: usize,
    pub training_batch_size: usize,
    pub training_epochs: usize,
    pub early_stopping_patience: usize,
    pub save_best_model: bool,
}

impl ModelTrainer {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let device = get_device();
        let var_store = nn::VarStore::new(device);
        let model = config.architecture.create_model(&var_store.root())?;
        let optimizer = nn::Adam::default().build(&var_store, config.learning_rate)?;

        Ok(Self {
            config,
            device,
            var_store,
            model,
            optimizer,
            best_validation_loss: f64::INFINITY,
            patience_counter: 0,
        })
    }

    pub fn train_epoch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>], batch_size: usize) -> Result<f64> {
        if features.len() != labels.len() {
            return Err(MLConfigError::InvalidConfig(format!(
                "Feature and label count mismatch: features {}, labels {}",
                features.len(),
                labels.len()
            )).into());
        }

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
        
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
        
        Ok(loss.double_value(&[]) as f64)
    }

    pub async fn train(
        &mut self,
        training_data: &[MarketData],
        validation_data: &[MarketData],
        early_stopping_patience: Option<usize>,
    ) -> Result<()> {
        let patience = early_stopping_patience.unwrap_or(self.config.early_stopping_patience);
        
        // --- Prepare Features and Labels --- 
        // TODO: Make horizon and threshold configurable
        let prediction_horizon = 60; // e.g., 1 hour ahead if data is 1-minute frequency
        let threshold = 0.005; // e.g., 0.5% change
        let min_history = self.config.window_size; // Or use a dedicated min_data_points config?
        // Assuming ModelConfig should have min_data_points required by preprocessor.
        // Let's add it to the local ModelConfig struct if it doesn't exist or use window_size.
        // For now, assume window_size is sufficient history for the preprocessor.
        // Rerun search for ModelConfig definition in this file if needed.

        info!("Preparing training features and labels...");
        let (train_features, train_labels) = prepare_features_and_labels(
            training_data,
            min_history, // Ensure this value is >= 26 for MACD
            prediction_horizon, 
            threshold
        )?;
        info!("Prepared {} training samples.", train_features.len());

        info!("Preparing validation features and labels...");
        let (val_features, val_labels) = prepare_features_and_labels(
            validation_data,
            min_history, // Ensure this value is >= 26 for MACD
            prediction_horizon, 
            threshold
        )?;
        info!("Prepared {} validation samples.", val_features.len());

        if train_features.is_empty() || val_features.is_empty() {
            warn!("No training or validation samples generated after preparation. Check data length, horizon, and threshold.");
            return Err(MLConfigError::InvalidConfig("No samples generated for training/validation".to_string()).into());
        }
        // --- End Preparation --- 

        for epoch in 0..self.config.training_epochs {
            // Training phase - Use prepared features/labels
            let training_loss = self.train_epoch(
                &train_features, 
                &train_labels,
                self.config.training_batch_size,
            )?;

            // Validation phase - Use prepared features/labels
            let validation_loss = self.evaluate(
                &val_features, 
                &val_labels,
            )?;

            info!(
                "Epoch {}: training_loss={:.4}, validation_loss={:.4}",
                epoch, training_loss, validation_loss
            );

            // Early stopping check
            if validation_loss < self.best_validation_loss {
                self.best_validation_loss = validation_loss;
                self.patience_counter = 0;
                
                if self.config.save_best_model {
                    self.save(Path::new(&self.config.model_path))?;
                    info!("Saved new best model (val_loss: {:.4}) to {}", validation_loss, self.config.model_path);
                }
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= patience {
                    warn!("Early stopping triggered after {} epochs", epoch + 1);
                    break;
                }
            }
        }

        Ok(())
    }

    fn evaluate(&self, features: &[Vec<f64>], labels: &[Vec<f64>]) -> Result<f64> {
        let batch_size = features.len() as i64;
        let feature_size = features[0].len() as i64;
        let label_size = labels[0].len() as i64;

        let input = Tensor::of_slice(&features.iter().flatten().cloned().collect::<Vec<_>>())
            .reshape(&[batch_size, feature_size])
            .to(self.device);
        
        let target = Tensor::of_slice(&labels.iter().flatten().cloned().collect::<Vec<_>>())
            .reshape(&[batch_size, label_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let loss = match self.config.loss_function {
            LossFunction::MSE => output.mse_loss(&target, tch::Reduction::Mean),
            LossFunction::CrossEntropy => output.cross_entropy_for_logits(&target),
        };
        
        Ok(loss.double_value(&[]) as f64)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.var_store.save(path)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        self.var_store.load(path)?;
        Ok(())
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
    fn test_model_trainer() -> Result<()> {
        let config = ModelConfig {
            architecture: ModelArchitecture::new(9, 20, 2),
            loss_function: LossFunction::CrossEntropy,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            window_size: 10,
            training_batch_size: 32,
            training_epochs: 10,
            early_stopping_patience: 5,
            save_best_model: true,
        };

        let mut trainer = ModelTrainer::new(config)?;
        
        // Create test data
        let mut training_data = Vec::new();
        let mut validation_data = Vec::new();
        
        for i in 0..100 {
            let data = create_test_market_data(
                100.0 + (i as f64 * 0.1),
                1000.0 + (i as f64 * 10.0),
                1_000_000_000.0 + (i as f64 * 10_000_000.0),
            );
            
            if i < 80 {
                training_data.push(data);
            } else {
                validation_data.push(data);
            }
        }

        // Train model
        trainer.train(&training_data, &validation_data, Some(5))?;
        
        Ok(())
    }
} 
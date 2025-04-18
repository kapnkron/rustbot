use crate::error::Result;
use crate::api::MarketData;
use crate::config::MLConfig;
use tch::{Device, Kind, Tensor, nn};
use std::path::Path;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};

pub struct ModelTrainer {
    config: MLConfig,
    device: Device,
    var_store: nn::VarStore,
    model: nn::Sequential,
    best_validation_loss: f64,
    patience_counter: usize,
}

impl ModelTrainer {
    pub fn new(config: MLConfig) -> Result<Self> {
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
            best_validation_loss: f64::INFINITY,
            patience_counter: 0,
        })
    }

    pub fn train_epoch(
        &mut self,
        features: &[Vec<f64>],
        labels: &[Vec<f64>],
        batch_size: usize,
    ) -> Result<f64> {
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

    fn train_batch(&mut self, features: &[Vec<f64>], labels: &[Vec<f64>]) -> Result<f64> {
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

    pub async fn train(
        &mut self,
        training_data: &[MarketData],
        validation_data: &[MarketData],
        early_stopping_patience: Option<usize>,
    ) -> Result<()> {
        let mut optimizer = nn::Adam::default().build(
            &self.var_store,
            self.config.learning_rate,
        )?;

        let patience = early_stopping_patience.unwrap_or(self.config.early_stopping_patience);
        
        for epoch in 0..self.config.training_epochs {
            // Training phase
            let training_loss = self.train_epoch(
                &training_data.iter().map(|d| d.to_features()).collect::<Vec<_>>(),
                &training_data.iter().map(|d| d.to_labels()).collect::<Vec<_>>(),
                self.config.training_batch_size,
            )?;

            // Validation phase
            let validation_loss = self.evaluate(
                &validation_data.iter().map(|d| d.to_features()).collect::<Vec<_>>(),
                &validation_data.iter().map(|d| d.to_labels()).collect::<Vec<_>>(),
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
                }
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= patience {
                    warn!("Early stopping triggered after {} epochs", epoch);
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

        let input = Tensor::of_slice(&features.concat())
            .reshape(&[batch_size, feature_size])
            .to(self.device);
        
        let target = Tensor::of_slice(&labels.concat())
            .reshape(&[batch_size, label_size])
            .to(self.device);
        
        let output = self.model.forward(&input);
        let loss = output.cross_entropy_for_logits(&target);
        
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
        let config = MLConfig {
            input_size: 9,
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
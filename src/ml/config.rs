use serde::{Deserialize, Serialize};
use thiserror::Error;
use super::architecture::{ModelArchitecture, LossFunction};

#[derive(Error, Debug)]
pub enum MLConfigError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Invalid model path: {0}")]
    InvalidModelPath(String),
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f64),
    #[error("Invalid window size: {0}")]
    InvalidWindowSize(usize),
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),
    #[error("Invalid number of epochs: {0}")]
    InvalidEpochs(usize),
    #[error("Invalid early stopping patience: {0}")]
    InvalidPatience(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub architecture: ModelArchitecture,
    pub loss_function: LossFunction,
    pub learning_rate: f64,
    pub model_path: String,
    pub window_size: usize,
    pub training_batch_size: usize,
    pub training_epochs: usize,
    pub early_stopping_patience: usize,
    pub save_best_model: bool,
    pub min_data_points: usize,
    pub validation_split: f64,
    pub evaluation_window_size: usize,
}

impl MLConfig {
    pub fn new(
        architecture: ModelArchitecture,
        loss_function: LossFunction,
        learning_rate: f64,
        model_path: String,
        window_size: usize,
        training_batch_size: usize,
        training_epochs: usize,
        early_stopping_patience: usize,
        save_best_model: bool,
        min_data_points: usize,
        validation_split: f64,
        evaluation_window_size: usize,
    ) -> Result<Self, MLConfigError> {
        // Validate learning rate
        if learning_rate <= 0.0 || learning_rate >= 1.0 {
            return Err(MLConfigError::InvalidLearningRate(learning_rate));
        }

        // Validate window size
        if window_size < 1 {
            return Err(MLConfigError::InvalidWindowSize(window_size));
        }

        // Validate batch size
        if training_batch_size < 1 {
            return Err(MLConfigError::InvalidBatchSize(training_batch_size));
        }

        // Validate epochs
        if training_epochs < 1 {
            return Err(MLConfigError::InvalidEpochs(training_epochs));
        }

        // Validate patience
        if early_stopping_patience < 1 {
            return Err(MLConfigError::InvalidPatience(early_stopping_patience));
        }

        // Validate validation split
        if validation_split <= 0.0 || validation_split >= 1.0 {
            return Err(MLConfigError::InvalidConfig(format!(
                "Validation split must be between 0 and 1, got {}",
                validation_split
            )));
        }

        // Validate model path
        if model_path.is_empty() {
            return Err(MLConfigError::InvalidModelPath(model_path));
        }

        Ok(Self {
            architecture,
            loss_function,
            learning_rate,
            model_path,
            window_size,
            training_batch_size,
            training_epochs,
            early_stopping_patience,
            save_best_model,
            min_data_points,
            validation_split,
            evaluation_window_size,
        })
    }

    pub fn validate_training_data(&self, data_len: usize) -> Result<(), MLConfigError> {
        if data_len < self.min_data_points {
            return Err(MLConfigError::InvalidConfig(format!(
                "Insufficient data points: got {}, required {}",
                data_len, self.min_data_points
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub loss_function: LossFunction,
    pub learning_rate: f64,
    pub model_path: String,
    pub window_size: usize,
    pub min_data_points: usize,
}

impl ModelConfig {
    pub fn new(
        architecture: ModelArchitecture,
        loss_function: LossFunction,
        learning_rate: f64,
        model_path: String,
        window_size: usize,
        min_data_points: usize,
    ) -> Result<Self, MLConfigError> {
        // Validate learning rate
        if learning_rate <= 0.0 || learning_rate >= 1.0 {
            return Err(MLConfigError::InvalidLearningRate(learning_rate));
        }

        // Validate window size
        if window_size < 1 {
            return Err(MLConfigError::InvalidWindowSize(window_size));
        }

        // Validate min data points
        if min_data_points < window_size {
            return Err(MLConfigError::InvalidConfig(format!(
                "Minimum data points must be at least window size, got {}",
                min_data_points
            )));
        }

        // Validate model path
        if model_path.is_empty() {
            return Err(MLConfigError::InvalidModelPath(model_path));
        }

        Ok(Self {
            architecture,
            loss_function,
            learning_rate,
            model_path,
            window_size,
            min_data_points,
        })
    }

    pub fn validate_training_data(&self, data_len: usize) -> Result<(), MLConfigError> {
        if data_len < self.min_data_points {
            return Err(MLConfigError::InvalidConfig(format!(
                "Insufficient data points: got {}, required {}",
                data_len, self.min_data_points
            )));
        }
        Ok(())
    }
} 
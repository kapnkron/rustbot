/// Contains the definition of the model architecture, activation functions, and loss functions.
// use tch::{Device, nn}; // Removed tch
use serde::{Deserialize, Serialize};
// use crate::error::Result; // Result is not used if create_model is removed

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub input_size: i64, // These fields can still be useful for config
    pub hidden_size: i64,
    pub output_size: i64,
    pub num_layers: usize,
    pub dropout: Option<f64>,
    pub activation: Activation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
}

impl ModelArchitecture {
    pub fn new(
        input_size: i64,
        hidden_size: i64,
        output_size: i64,
        num_layers: usize,
        dropout: Option<f64>,
        activation: Activation,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_layers,
            dropout,
            activation,
        }
    }

    /* // Commenting out tch-dependent create_model
    pub fn create_model(&self, vs: &nn::Path) -> Result<nn::Sequential> {
        let mut seq = nn::seq();
        let activation = self.activation.clone();
        
        // Input layer
        seq = seq.add(nn::linear(
            vs,
            self.input_size,
            self.hidden_size,
            Default::default(),
        ));
        
        // Hidden layers
        for _ in 1..self.num_layers {
            let activation = activation.clone();
            seq = seq.add_fn(move |xs| match activation {
                Activation::ReLU => xs.relu(),
                Activation::Tanh => xs.tanh(),
                Activation::Sigmoid => xs.sigmoid(),
            });
            
            if let Some(dropout) = self.dropout {
                seq = seq.add_fn(move |xs| xs.dropout(dropout, true));
            }
            
            seq = seq.add(nn::linear(
                vs,
                self.hidden_size,
                self.hidden_size,
                Default::default(),
            ));
        }
        
        // Output layer
        seq = seq.add(nn::linear(
            vs,
            self.hidden_size,
            self.output_size,
            Default::default(),
        ));
        
        Ok(seq)
    }
    */
}

/* // Commenting out tch-dependent get_device
/// Attempts to get the CUDA device if available, otherwise falls back to CPU.
pub fn get_device() -> Device {
    // Rely solely on tch's built-in check, which depends on build-time detection
    if tch::utils::has_cuda() {
        log::info!("CUDA detected by tch, attempting to use GPU.");
        Device::cuda_if_available()
    } else {
        log::info!("CUDA not detected by tch or build script, using CPU.");
        Device::Cpu
    }
}
*/ 
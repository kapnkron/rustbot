use tch::{Device, nn};
use serde::{Deserialize, Serialize};
use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub input_size: i64,
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
}

pub fn get_device() -> Device {
    // Only check for CUDA if the "cuda" feature is enabled AND CUDA is available
    if cfg!(feature = "cuda") && tch::Cuda::is_available() {
        log::info!("CUDA feature enabled and CUDA device found. Using CUDA:0.");
        Device::Cuda(0)
    } else {
        log::info!("CUDA not available or feature not enabled. Using CPU.");
        Device::Cpu
    }
} 
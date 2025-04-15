use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub api: ApiConfig,
    pub trading: TradingConfig,
    pub monitoring: MonitoringConfig,
    pub telegram: TelegramConfig,
    pub database: DatabaseConfig,
    pub security: SecurityConfig,
    pub ml: MLConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiConfig {
    pub coingecko_api_key: String,
    pub coinmarketcap_api_key: String,
    pub cryptodatadownload_api_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TradingConfig {
    pub risk_level: f64,
    pub max_position_size: f64,
    pub stop_loss_percentage: f64,
    pub take_profit_percentage: f64,
    pub trading_pairs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_prometheus: bool,
    pub prometheus_port: u16,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub price_change_threshold: f64,
    pub volume_threshold: f64,
    pub error_rate_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub bot_token: String,
    pub chat_id: String,
    pub enable_notifications: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_2fa: bool,
    pub api_key_rotation_days: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLConfig {
    pub input_size: i64,
    pub hidden_size: i64,
    pub output_size: i64,
    pub learning_rate: f64,
    pub model_path: String,
    pub confidence_threshold: f64,
    pub training_batch_size: usize,
    pub training_epochs: usize,
    pub window_size: usize,
    pub min_data_points: usize,
    pub validation_split: f64,
    pub early_stopping_patience: usize,
    pub save_best_model: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let config_str = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let config_str = toml::to_string_pretty(self)?;
        fs::write(path, config_str)?;
        Ok(())
    }
} 
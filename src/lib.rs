pub mod api;
pub mod bot;
pub mod config;
pub mod error;
pub mod logging;
pub mod metrics;
pub mod models;
pub mod strategies;
pub mod utils;
pub mod validation;
pub mod trading;
pub mod ml;
pub mod monitoring;
pub mod security;
pub mod telegram;
pub mod wallet;
pub mod services;
pub mod solana;
pub mod cli;
pub mod data_loader;

pub use error::{Error, Result};

// Declare tests module only when testing
#[cfg(test)]
pub mod tests; 

// Re-export key types for convenience
pub use config::Config;
pub use trading::TradingBot;
pub use telegram::TelegramBot;
pub use api::MarketDataProvider;
pub use data_loader::load_market_data_from_csv; 
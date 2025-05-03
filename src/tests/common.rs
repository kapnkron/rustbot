#[cfg(test)]
use super::super::config::{Config, ApiConfig, TradingConfig, MonitoringConfig, AlertThresholds, TelegramConfig, DatabaseConfig, SecurityConfig, MLConfig, SolanaConfig, DexTradingConfig};
use super::super::ml::{ModelArchitecture, Activation, LossFunction, ModelConfig};
use super::super::trading::{TradingMarketData, Trade, Position, TradingSignal, SignalType};
use super::super::api::types; // Import for types::Quote, types::USDData
use chrono::{Utc, TimeZone};
use super::super::ml::Predictor;
use super::super::error::Result;
use async_trait::async_trait;
use log;

// Helper to create a default test config
pub fn create_test_config() -> Config {
    Config {
        api: ApiConfig {
            coingecko_api_key: "test".to_string(),
            coinmarketcap_api_key: "test".to_string(),
            cryptodatadownload_api_key: "test".to_string(),
        },
        trading: TradingConfig {
            risk_level: 0.5,
            max_position_size: 1000.0,
            stop_loss_percentage: 0.02,
            take_profit_percentage: 0.1,
            trading_pairs: vec!["SOL/USDC".to_string()],
        },
        monitoring: MonitoringConfig {
            enable_prometheus: false,
            prometheus_port: 9090,
            alert_thresholds: AlertThresholds {
                price_change_threshold: 0.1,
                volume_threshold: 1000.0,
                error_rate_threshold: 0.05,
            },
        },
        telegram: TelegramConfig {
            bot_token: "test".to_string(),
            chat_id: "test".to_string(),
            enable_notifications: true,
        },
        database: DatabaseConfig {
            url: "test".to_string(),
            max_connections: 10,
        },
        security: SecurityConfig {
            api_key_rotation_days: 30,
            keychain_service_name: "test-bot".to_string(),
            solana_key_username: "test-sol-key".to_string(),
            ton_key_username: "test-ton-key".to_string(),
        },
        ml: MLConfig {
            architecture: ModelArchitecture {
                input_size: 9, hidden_size: 20, output_size: 2,
                num_layers: 2, dropout: Some(0.1), activation: Activation::ReLU,
            },
            loss_function: LossFunction::MSE,
            input_size: 9, // Should match architecture
            hidden_size: 20, // Should match architecture
            output_size: 2, // Should match architecture
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            confidence_threshold: 0.7,
            training_batch_size: 32,
            training_epochs: 100,
            window_size: 10, // Reasonable window size
            min_data_points: 100, // <-- Increased to ensure enough history for indicators
            validation_split: 0.2,
            early_stopping_patience: 5,
            save_best_model: true,
            evaluation_window_size: 100,
        },
        solana: SolanaConfig {
            rpc_url: "http://localhost:8899".to_string(),
        },
        dex_trading: DexTradingConfig {
            trading_pair_symbol: "SOL/USDC".to_string(),
            base_token_mint: "So11111111111111111111111111111111111111112".to_string(),
            quote_token_mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(),
            base_token_decimals: 9,
            quote_token_decimals: 6,
            slippage_bps: 50,
        },
    }
}

// Helper to create test market data
pub fn create_test_trading_market_data(symbol: &str, price: f64) -> TradingMarketData {
    TradingMarketData {
        symbol: symbol.to_string(),
        price,
        volume: 1000.0,
        market_cap: 1_000_000.0,
        price_change_24h: 0.0,
        volume_change_24h: 0.0,
        timestamp: Utc::now(),
        volume_24h: 1000.0,
        change_24h: 0.0,
        quote: types::Quote {
            usd: types::USDData {
                price,
                volume_24h: 1000.0,
                market_cap: 1_000_000.0,
                percent_change_24h: 0.0,
                volume_change_24h: 0.0,
            }
        }
    }
}

// Added MockMarketDataCollector definition
#[derive(Clone)] // Add Clone
pub struct MockMarketDataCollector;

impl Default for MockMarketDataCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MockMarketDataCollector {
    pub fn new() -> Self { Self }
}

// Dummy implementation (adjust as needed for monitoring tests)
#[async_trait]
impl super::super::api::MarketDataProvider for MockMarketDataCollector {
    async fn collect_market_data(&mut self, symbol: &str) -> Result<types::MarketData> {
        // Return some default MarketData
        Ok(types::MarketData {
            timestamp: Utc::now(),
            symbol: symbol.to_string(),
            price: 50.0,
            volume: 1000.0,
            market_cap: 1000000.0,
            price_change_24h: 0.1,
            volume_change_24h: 0.2,
            volume_24h: 1200.0,
            change_24h: 0.1,
            quote: types::Quote { usd: types::USDData { price: 50.0, volume_24h: 1000.0, market_cap: 1000000.0, percent_change_24h: 0.1, volume_change_24h: 0.2 } }, // Populate Quote as needed
        })
    }
}

// Function to create the mock collector easily
pub fn create_mock_market_data_collector() -> MockMarketDataCollector {
    MockMarketDataCollector::new()
}

// Dummy Predictor for testing
#[derive(Debug)]
pub struct DummyPredictor;

#[async_trait]
impl Predictor for DummyPredictor {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> { 
        // Added logging
        log::debug!(
            "DummyPredictor received data with price: {:.2}",
            data.price
        );

        let result = if data.price > 105.0 {
            Ok(vec![0.9, 0.1])
        } else if data.price < 95.0 {
            Ok(vec![0.1, 0.9])
        } else {
            Ok(vec![0.5, 0.5])
        };

        // Added logging
        log::debug!("DummyPredictor returning: {:?}", result);
        result
    }
}

// Helper to create a default ModelConfig for testing
pub fn create_test_model_config() -> ModelConfig {
    ModelConfig {
        architecture: ModelArchitecture {
            input_size: 11, // Changed from 9 to 11 to match preprocessor output
            hidden_size: 20,
            output_size: 2,
            num_layers: 2,
            dropout: Some(0.1),
            activation: Activation::ReLU,
        },
        loss_function: LossFunction::MSE,
        learning_rate: 0.001,
        model_path: "test_model.pt".to_string(),
        window_size: 10,
        min_data_points: 100,
    }
}

// Helper function to create a default Trade for testing
pub fn create_test_trade(symbol: &str, pnl: f64) -> Trade {
    Trade {
        entry_time: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
        exit_time: Utc.with_ymd_and_hms(2024, 1, 1, 13, 0, 0).unwrap(),
        symbol: symbol.to_string(),
        entry_price: 100.0,
        exit_price: if pnl > 0.0 { 110.0 } else { 90.0 },
        size: 1.0, 
        pnl,
        pnl_percentage: pnl / 100.0,
    }
}

// Helper function to create a default Position for testing
pub fn create_test_position(symbol: &str, entry_price: f64, amount: f64) -> Position {
    Position {
        symbol: symbol.to_string(),
        amount,
        entry_price,
        current_price: entry_price * 1.05,
        unrealized_pnl: amount * (entry_price * 0.05),
        size: amount,
        entry_time: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
    }
}

// Helper function to create a default TradingSignal for testing
pub fn create_test_signal(symbol: &str, signal_type: SignalType, price: f64, confidence: f64) -> TradingSignal {
    TradingSignal {
        symbol: symbol.to_string(),
        signal_type,
        price,
        confidence,
    }
} 
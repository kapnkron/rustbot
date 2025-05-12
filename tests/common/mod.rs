use trading_bot::config::{Config, ApiConfig, TradingConfig, MonitoringConfig, AlertThresholds, TelegramConfig, DatabaseConfig, SecurityConfig, MLConfig, SolanaConfig, DexTradingConfig};
use trading_bot::ml::{ModelArchitecture, Activation, LossFunction, ModelConfig};
use trading_bot::trading::TradingMarketData;
use trading_bot::api::types; // Import for types::Quote, types::USDData
use chrono::{Utc, DateTime, TimeZone, NaiveDate}; // Add TimeZone, NaiveDate
use trading_bot::ml::Predictor;
use trading_bot::error::Result;
use async_trait::async_trait;
use trading_bot::trading::{Trade, Position, TradingSignal, SignalType};
use std::fs::File;
use std::path::Path;
use csv;
use serde::Deserialize;

// Helper to create a default test config
// (Moved from src/trading/mod.rs)
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
            input_size: 9,
            hidden_size: 20,
            output_size: 2,
            learning_rate: 0.001,
            model_path: "model.pt".to_string(),
            confidence_threshold: 0.7,
            training_batch_size: 32,
            training_epochs: 100,
            window_size: 1,
            min_data_points: 1,
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
// (Moved from src/trading/mod.rs tests)
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

// Dummy Predictor for testing
// (Moved from src/trading/mod.rs tests)
#[derive(Debug)] // Add Debug derive
pub struct DummyPredictor;

#[async_trait]
impl Predictor for DummyPredictor {
    async fn predict(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> { 
        // Internal logic remains synchronous
        if data.price > 105.0 {
            Ok(vec![0.9, 0.1])
        } else if data.price < 95.0 {
            Ok(vec![0.1, 0.9])
        } else {
                Ok(vec![0.5, 0.5])
        }
    }
}

// Helper to create a default ModelConfig for testing
// (Based on setups in src/ml/mod.rs tests)
pub fn create_test_model_config() -> ModelConfig {
    ModelConfig {
        architecture: ModelArchitecture {
            input_size: 9, // Example value, adjust if needed
            hidden_size: 20,
            output_size: 2,
            num_layers: 2,
            dropout: Some(0.1),
            activation: Activation::ReLU,
        },
        loss_function: LossFunction::MSE,
        learning_rate: 0.001,
        model_path: "test_model.pt".to_string(), // Use a consistent test path maybe?
        window_size: 10, // Example value
        min_data_points: 100, // Example value
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
        pnl_percentage: pnl / 100.0, // Assuming entry size was 1 * 100.0
    }
}

// Helper function to create a default Position for testing
pub fn create_test_position(symbol: &str, entry_price: f64, amount: f64) -> Position {
    Position {
        symbol: symbol.to_string(),
        amount,
        entry_price,
        current_price: entry_price * 1.05, // Example: slightly profitable
        unrealized_pnl: amount * (entry_price * 0.05),
        size: amount * entry_price,
        entry_time: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
    }
}

// Helper function to create a default TradingSignal for testing
pub fn create_test_signal(symbol: &str, signal_type: SignalType, price: f64, confidence: f64) -> TradingSignal {
    TradingSignal {
        symbol: symbol.to_string(),
        signal_type,
        confidence,
        price,
    }
}

#[derive(Debug, Deserialize)]
struct MarketDataRecord {
    date: String, // Assuming ISO 8601 format like "2023-01-01T12:00:00Z", maps to CSV 'date'
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64, 
    // Add other direct CSV columns here if needed, e.g., market_cap_direct: Option<f64>
}

pub fn load_market_data_from_csv(file_path: &str, symbol: &str) -> Result<Vec<TradingMarketData>> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(trading_bot::error::Error::Generic(format!("File not found: {}", file_path)));
    }

    let file = File::open(path).map_err(|e| trading_bot::error::Error::Generic(format!("Failed to open file {}: {}", file_path, e)))?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut market_data_vec = Vec::new();

    for result in rdr.deserialize() {
        let record: MarketDataRecord = result.map_err(|e| trading_bot::error::Error::Generic(format!("Failed to deserialize CSV record: {}", e)))?;
        
        let naive_date = NaiveDate::parse_from_str(&record.date, "%Y-%m-%d")
            .map_err(|e| trading_bot::error::Error::Generic(format!("Failed to parse date string '{}': {}", record.date, e)))?;
        let naive_datetime = naive_date.and_hms_opt(0, 0, 0)
            .ok_or_else(|| trading_bot::error::Error::Generic(format!("Failed to create NaiveDateTime from date: {}", record.date)))?;
        let timestamp = Utc.from_utc_datetime(&naive_datetime);

        // Using 'close' for price, and direct 'volume'.
        // OHL are parsed but not directly mapped to TradingMarketData fields yet, as it doesn't have them.
        // We can extend TradingMarketData later if OHL are needed directly in Rust structs.
        let market_data = TradingMarketData {
            symbol: symbol.to_string(),
            price: record.close, // Using 'close' price from CSV
            volume: record.volume,
            market_cap: record.close * record.volume, // Example calculation, adjust if CSV has direct market_cap
            price_change_24h: 0.0, // Placeholder - can be calculated or read if CSV provides it
            volume_change_24h: 0.0, // Placeholder - can be calculated or read
            timestamp,
            volume_24h: record.volume, // Assuming CSV 'volume' is daily volume
            change_24h: 0.0, // Placeholder - can be calculated
            quote: types::Quote {
                usd: types::USDData {
                    price: record.close,
                    volume_24h: record.volume,
                    market_cap: record.close * record.volume, // As above
                    percent_change_24h: 0.0, // Placeholder
                    volume_change_24h: 0.0, // Placeholder
                }
            }
        };
        market_data_vec.push(market_data);
    }
    Ok(market_data_vec)
}

// New structs and function for loading FEATURED market data

// Private helper struct for direct deserialization of featured CSVs
#[derive(Debug, Deserialize)]
struct CsvFeaturedRecord {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(rename = "return")]
    return_val: f64,
    log_return: f64,
    sma_10: f64,
    ema_10: f64,
    sma_50: f64,
    ema_50: f64,
    rsi_14: f64,
    macd: f64,
    macd_signal: f64,
    macd_diff: f64,
    bb_high: f64,
    bb_low: f64,
    bb_width: f64,
    volatility_10: f64,
    volatility_50: f64,
    close_lag_1: f64,
    close_lag_2: f64,
    close_lag_3: f64,
}

// Public struct to hold the parsed featured market data for use in tests
#[derive(Debug)]
pub struct FeaturedMarketData {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub return_val: f64,
    pub log_return: f64,
    pub sma_10: f64,
    pub ema_10: f64,
    pub sma_50: f64,
    pub ema_50: f64,
    pub rsi_14: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_diff: f64,
    pub bb_high: f64,
    pub bb_low: f64,
    pub bb_width: f64,
    pub volatility_10: f64,
    pub volatility_50: f64,
    pub close_lag_1: f64,
    pub close_lag_2: f64,
    pub close_lag_3: f64,
}

pub fn load_featured_data_from_csv(file_path: &str) -> Result<Vec<FeaturedMarketData>> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(trading_bot::error::Error::Generic(format!("File not found: {}", file_path)));
    }

    let file = File::open(path).map_err(|e| trading_bot::error::Error::Generic(format!("Failed to open file {}: {}", file_path, e)))?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut featured_data_vec = Vec::new();

    for result in rdr.deserialize() {
        let csv_record: CsvFeaturedRecord = result.map_err(|e| trading_bot::error::Error::Generic(format!("Failed to deserialize CSV record for featured data: {}", e)))?;
        
        let naive_date = NaiveDate::parse_from_str(&csv_record.date, "%Y-%m-%d")
            .map_err(|e| trading_bot::error::Error::Generic(format!("Failed to parse date string '{}' for featured data: {}", csv_record.date, e)))?;
        let naive_datetime = naive_date.and_hms_opt(0,0,0)
            .ok_or_else(|| trading_bot::error::Error::Generic(format!("Failed to create NaiveDateTime from date for featured data: {}", csv_record.date)))?;
        let timestamp = Utc.from_utc_datetime(&naive_datetime);

        featured_data_vec.push(FeaturedMarketData {
            timestamp,
            open: csv_record.open,
            high: csv_record.high,
            low: csv_record.low,
            close: csv_record.close,
            return_val: csv_record.return_val,
            log_return: csv_record.log_return,
            sma_10: csv_record.sma_10,
            ema_10: csv_record.ema_10,
            sma_50: csv_record.sma_50,
            ema_50: csv_record.ema_50,
            rsi_14: csv_record.rsi_14,
            macd: csv_record.macd,
            macd_signal: csv_record.macd_signal,
            macd_diff: csv_record.macd_diff,
            bb_high: csv_record.bb_high,
            bb_low: csv_record.bb_low,
            bb_width: csv_record.bb_width,
            volatility_10: csv_record.volatility_10,
            volatility_50: csv_record.volatility_50,
            close_lag_1: csv_record.close_lag_1,
            close_lag_2: csv_record.close_lag_2,
            close_lag_3: csv_record.close_lag_3,
        });
    }
    Ok(featured_data_vec)
}

// Add other common test utilities here later

#[cfg(test)]
mod tests {
    use super::*; // Make items from parent module (common) available
    use chrono::{Datelike, Timelike};

    #[test]
    fn test_load_featured_data_successfully() {
        // Assuming 'data/features/wif_usd_features.csv' exists and is readable
        // You might want to create a smaller, dedicated test fixture CSV for this test
        // to make it more controlled and faster.
        // For now, we use an existing one as per our discussion.
        let file_path = "data/features/wif_usd_features.csv"; 
        
        let result = load_featured_data_from_csv(file_path);
        assert!(result.is_ok(), "Should load featured data successfully. Error: {:?}", result.err());

        let data_vec = result.unwrap();
        assert!(!data_vec.is_empty(), "Loaded data should not be empty.");

        // Check the first record based on the known content of wif_usd_features.csv
        // date,open,high,low,close,return,log_return,sma_10,ema_10,sma_50,ema_50,rsi_14,macd,macd_signal,macd_diff,bb_high,bb_low,bb_width,volatility_10,volatility_50,close_lag_1,close_lag_2,close_lag_3
        // 2024-11-25,3.06,3.56,3.0,3.17,0.035947712418300526,0.03531667192489972,2.741,2.829429232790332,2.3056,2.4149446527110214,58.977004035327624,0.2722294925363702,0.16143497094279222,0.11079452159357797,3.5475283455742668,1.298471654425733,2.2490566911485335,0.17818376295022564,0.16944403147355175,3.06,3.6,3.31

        if let Some(first_record) = data_vec.first() {
            assert_eq!(first_record.timestamp.year(), 2024);
            assert_eq!(first_record.timestamp.month(), 11);
            assert_eq!(first_record.timestamp.day(), 25);
            assert_eq!(first_record.timestamp.hour(), 0); // Assuming midnight UTC

            assert_eq!(first_record.open, 3.06);
            assert_eq!(first_record.high, 3.56);
            assert_eq!(first_record.low, 3.0);
            assert_eq!(first_record.close, 3.17);
            assert!((first_record.return_val - 0.0359477124).abs() < 1e-9);
            assert!((first_record.log_return - 0.0353166719).abs() < 1e-9);
            assert_eq!(first_record.sma_10, 2.741);
            // Add more assertions for other fields as desired for thoroughness
            assert_eq!(first_record.close_lag_1, 3.06);
            assert_eq!(first_record.close_lag_2, 3.6);
            assert_eq!(first_record.close_lag_3, 3.31);
        } else {
            panic!("Data vector was empty after successful load, which is unexpected.");
        }
    }

    #[test]
    fn test_load_featured_data_file_not_found() {
        let file_path = "data/features/non_existent_file.csv";
        let result = load_featured_data_from_csv(file_path);
        assert!(result.is_err(), "Should return an error for a non-existent file.");
        if let Err(e) = result {
            assert!(e.to_string().contains("File not found"), "Error message should indicate file not found.");
        }
    }
    
    #[test]
    fn test_load_raw_data_successfully() {
        let file_path = "tests/fixtures/sample_raw_ohlcv.csv";
        let symbol = "BTC/USD";
        
        let result = load_market_data_from_csv(file_path, symbol);
        assert!(result.is_ok(), "Should load raw data successfully. Error: {:?}", result.err());

        let data_vec = result.unwrap();
        assert_eq!(data_vec.len(), 3, "Should load 3 records from the sample raw CSV.");

        // Check the first record
        if let Some(first_record) = data_vec.first() {
            assert_eq!(first_record.symbol, symbol);
            assert_eq!(first_record.timestamp.year(), 2023);
            assert_eq!(first_record.timestamp.month(), 1);
            assert_eq!(first_record.timestamp.day(), 1);
            assert_eq!(first_record.timestamp.hour(), 0); // Assuming midnight UTC

            assert_eq!(first_record.price, 102.5); // From 'close' column
            assert_eq!(first_record.volume, 1000.0);
            
            // Check calculated market_cap (close * volume)
            assert_eq!(first_record.market_cap, 102.5 * 1000.0);
            
            // Check placeholder values for other fields
            assert_eq!(first_record.price_change_24h, 0.0);
            assert_eq!(first_record.volume_change_24h, 0.0);
            assert_eq!(first_record.volume_24h, 1000.0); // Mapped from 'volume'
            assert_eq!(first_record.change_24h, 0.0);

            assert_eq!(first_record.quote.usd.price, 102.5);
            assert_eq!(first_record.quote.usd.volume_24h, 1000.0);
            assert_eq!(first_record.quote.usd.market_cap, 102.5 * 1000.0);
            assert_eq!(first_record.quote.usd.percent_change_24h, 0.0);
            assert_eq!(first_record.quote.usd.volume_change_24h, 0.0);
        } else {
            panic!("Data vector was empty after successful load of raw data, which is unexpected.");
        }
    }

    #[test]
    fn test_load_raw_data_file_not_found() {
        let file_path = "tests/fixtures/non_existent_raw.csv";
        let symbol = "BTC/USD";
        let result = load_market_data_from_csv(file_path, symbol);
        assert!(result.is_err(), "Should return an error for a non-existent raw data file.");
        if let Err(e) = result {
            assert!(e.to_string().contains("File not found"), "Error message for raw data should indicate file not found.");
        }
    }
} 
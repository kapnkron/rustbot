use crate::error::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::ml::TradingModel;
use crate::monitoring::Monitor;
use crate::api::MarketDataCollector;
use log::{info, error};
use crate::api::types;

mod risk;
pub use risk::RiskManager;

mod backtest;
pub use backtest::{Backtester, BacktestResult, Trade};

#[derive(Debug, Clone)]
pub struct TradingMarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub volume_24h: f64,
    pub change_24h: f64,
    pub quote: types::Quote,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub amount: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub size: f64,
    pub entry_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone)]
pub struct TradingBot {
    market_data_collector: Arc<Mutex<MarketDataCollector>>,
    positions: Arc<Mutex<Vec<Position>>>,
    risk_level: Arc<Mutex<f64>>,
    trading_enabled: Arc<Mutex<bool>>,
}

impl TradingBot {
    pub fn new(market_data_collector: MarketDataCollector) -> Self {
        Self {
            market_data_collector: Arc::new(Mutex::new(market_data_collector)),
            positions: Arc::new(Mutex::new(Vec::new())),
            risk_level: Arc::new(Mutex::new(0.5)),
            trading_enabled: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<TradingMarketData> {
        let data = self.market_data_collector.lock().await.collect_market_data(symbol).await?;
        Ok(data.into())
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(self.positions.lock().await.clone())
    }

    pub async fn set_risk_level(&self, level: f64) -> Result<()> {
        if level >= 0.0 && level <= 1.0 {
            *self.risk_level.lock().await = level;
            info!("Risk level set to {}", level);
            Ok(())
        } else {
            Err(crate::error::Error::ValidationError(
                "Risk level must be between 0.0 and 1.0".to_string(),
            ))
        }
    }

    pub async fn process_market_data(&self, data: TradingMarketData) -> Result<Option<TradingSignal>> {
        if !*self.trading_enabled.lock().await {
            return Ok(None);
        }

        // Simple trading strategy based on price and volume changes
        let signal = if data.price_change_24h > 5.0 && data.volume_change_24h > 20.0 {
            Some(TradingSignal {
                symbol: data.symbol.clone(),
                signal_type: SignalType::Buy,
                confidence: 0.7,
                price: data.price,
            })
        } else if data.price_change_24h < -5.0 && data.volume_change_24h > 20.0 {
            Some(TradingSignal {
                symbol: data.symbol.clone(),
                signal_type: SignalType::Sell,
                confidence: 0.7,
                price: data.price,
            })
        } else {
            None
        };

        if let Some(ref signal) = signal {
            info!("Generated trading signal: {:?}", signal);
        }

        Ok(signal)
    }

    pub async fn execute_signal(&self, signal: TradingSignal) -> Result<()> {
        if !*self.trading_enabled.lock().await {
            return Ok(());
        }

        let risk_level = *self.risk_level.lock().await;
        let mut positions = self.positions.lock().await;

        match signal.signal_type {
            SignalType::Buy => {
                // Calculate position size based on risk level
                let position_size = 1000.0 * risk_level; // Example: $1000 base position size
                
                positions.push(Position {
                    symbol: signal.symbol.clone(),
                    amount: position_size / signal.price,
                    entry_price: signal.price,
                    current_price: signal.price,
                    unrealized_pnl: 0.0,
                    size: position_size,
                    entry_time: Utc::now(),
                });
                
                info!("Opened long position for {} at ${}", signal.symbol, signal.price);
            }
            SignalType::Sell => {
                // Close any existing positions
                positions.retain(|p| p.symbol != signal.symbol);
                info!("Closed position for {} at ${}", signal.symbol, signal.price);
            }
            SignalType::Hold => {
                // Update existing positions
                for position in positions.iter_mut() {
                    if position.symbol == signal.symbol {
                        position.current_price = signal.price;
                        position.unrealized_pnl = (signal.price - position.entry_price) * position.amount;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_config() -> crate::config::Config {
        crate::config::Config {
            api: crate::config::ApiConfig {
                coingecko_api_key: "test".to_string(),
                coinmarketcap_api_key: "test".to_string(),
                cryptodatadownload_api_key: "test".to_string(),
            },
            trading: crate::config::TradingConfig {
                risk_level: 0.5,
                max_position_size: 1000.0,
                stop_loss_percentage: 0.02,
                take_profit_percentage: 0.1,
                trading_pairs: vec!["BTC/USD".to_string()],
            },
            monitoring: crate::config::MonitoringConfig {
                enable_prometheus: false,
                prometheus_port: 9090,
                alert_thresholds: crate::config::AlertThresholds {
                    price_change_threshold: 0.1,
                    volume_threshold: 1000.0,
                    error_rate_threshold: 0.05,
                },
            },
            telegram: crate::config::TelegramConfig {
                bot_token: "test".to_string(),
                chat_id: "test".to_string(),
                enable_notifications: true,
            },
            database: crate::config::DatabaseConfig {
                url: "test".to_string(),
                max_connections: 10,
            },
            security: crate::config::SecurityConfig {
                enable_2fa: false,
                api_key_rotation_days: 30,
            },
            ml: crate::config::MLConfig {
                input_size: 9,
                hidden_size: 20,
                output_size: 2,
                learning_rate: 0.001,
                model_path: "model.pt".to_string(),
                confidence_threshold: 0.7,
                training_batch_size: 32,
                training_epochs: 100,
                window_size: 10,
                min_data_points: 100,
                validation_split: 0.2,
                early_stopping_patience: 5,
                save_best_model: true,
                evaluation_window_size: 100,
            },
        }
    }

    #[tokio::test]
    async fn test_position_management() {
        let config = create_test_config();
        let market_data_collector = MarketDataCollector::new(
            config.api.coingecko_api_key.clone(),
            config.api.coinmarketcap_api_key.clone(),
            config.api.cryptodatadownload_api_key.clone(),
        );
        let mut bot = TradingBot::new(market_data_collector);

        // Test opening position
        let signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 100.0,
            price: 100.0,
        };

        assert!(bot.execute_signal(signal).await.is_ok());
    }
} 
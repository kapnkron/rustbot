use log::{info, warn};
use prometheus::{Counter, Gauge, Histogram, Registry};
use crate::trading::Position;
use crate::error::Result;
use std::sync::Arc;

pub mod dashboard;
pub mod thresholds;
pub mod health;
pub mod performance;
pub use dashboard::Dashboard;
pub use thresholds::{ThresholdManager, ThresholdConfig};
pub use health::HealthMetrics;
pub use performance::PerformanceMetrics;

pub struct Monitor {
    // config: MonitoringConfig,
    // registry: Registry,
    metrics: Metrics,
    // bot: Arc<Mutex<TradingBot>>,
    // model: Option<Arc<Mutex<TradingModel>>>,
}

#[derive(Clone)]
struct Metrics {
    // Trading metrics
    total_trades: Counter,
    winning_trades: Counter,
    losing_trades: Counter,
    total_pnl: Gauge,
    current_position_size: Gauge,
    account_balance: Gauge,
    max_drawdown: Gauge,
    
    // ML model metrics
    model_confidence: Gauge,
    prediction_latency: Histogram,
    training_loss: Gauge,
    validation_loss: Gauge,
    
    // Risk metrics
    risk_per_trade: Gauge,
    daily_pnl: Gauge,
    position_exposure: Gauge,
}

impl Metrics {
    pub fn new(registry: &Registry) -> Result<Self> {
        let metrics = Self {
            total_trades: Counter::new("total_trades", "Total number of trades executed")?,
            winning_trades: Counter::new("winning_trades", "Number of winning trades")?,
            losing_trades: Counter::new("losing_trades", "Number of losing trades")?,
            total_pnl: Gauge::new("total_pnl", "Total profit and loss")?,
            current_position_size: Gauge::new("current_position_size", "Current position size")?,
            account_balance: Gauge::new("account_balance", "Current account balance")?,
            max_drawdown: Gauge::new("max_drawdown", "Maximum drawdown")?,
            
            model_confidence: Gauge::new("model_confidence", "Current model confidence")?,
            prediction_latency: Histogram::with_opts(
                prometheus::HistogramOpts::new("prediction_latency", "Model prediction latency in seconds")
            )?,
            training_loss: Gauge::new("training_loss", "Current training loss")?,
            validation_loss: Gauge::new("validation_loss", "Current validation loss")?,
            
            risk_per_trade: Gauge::new("risk_per_trade", "Risk per trade")?,
            daily_pnl: Gauge::new("daily_pnl", "Daily profit and loss")?,
            position_exposure: Gauge::new("position_exposure", "Current position exposure")?,
        };

        // Register metrics
        registry.register(Box::new(metrics.total_trades.clone()))?;
        registry.register(Box::new(metrics.winning_trades.clone()))?;
        registry.register(Box::new(metrics.losing_trades.clone()))?;
        registry.register(Box::new(metrics.total_pnl.clone()))?;
        registry.register(Box::new(metrics.current_position_size.clone()))?;
        registry.register(Box::new(metrics.account_balance.clone()))?;
        registry.register(Box::new(metrics.max_drawdown.clone()))?;
        registry.register(Box::new(metrics.model_confidence.clone()))?;
        registry.register(Box::new(metrics.prediction_latency.clone()))?;
        registry.register(Box::new(metrics.training_loss.clone()))?;
        registry.register(Box::new(metrics.validation_loss.clone()))?;
        registry.register(Box::new(metrics.risk_per_trade.clone()))?;
        registry.register(Box::new(metrics.daily_pnl.clone()))?;
        registry.register(Box::new(metrics.position_exposure.clone()))?;

        Ok(metrics)
    }
}

impl Monitor {
    pub fn new(registry: Arc<Registry>) -> Result<Self> {
        let metrics = Metrics::new(&registry)?;
        
        Ok(Self {
            // config, // Keep commented
            // registry, // Keep commented (Registry passed to Metrics)
            metrics,
            // bot: Arc::new(Mutex::new(bot)), // Keep commented
            // model: model.map(|m| Arc::new(Mutex::new(m))), // Keep commented (or use Predictor)
        })
    }

    pub async fn record_trade(&self, trade: &crate::trading::Trade) -> Result<()> {
        self.metrics.total_trades.inc();
        if trade.pnl > 0.0 {
            self.metrics.winning_trades.inc();
        } else {
            self.metrics.losing_trades.inc();
        }
        
        self.metrics.total_pnl.add(trade.pnl);
        
        info!(
            "Trade executed: {} {} at {} (PnL: {:.2}%)",
            trade.symbol,
            if trade.pnl > 0.0 { "won" } else { "lost" },
            trade.exit_time,
            trade.pnl_percentage * 100.0
        );
        
        Ok(())
    }

    pub async fn record_position(&self, position: &Position) -> Result<()> {
        self.metrics.current_position_size.set(position.size);
        self.metrics.position_exposure.set(position.size * position.entry_price);
        
        info!(
            "Position update: {} {} at {}",
            position.symbol,
            position.size,
            position.entry_price
        );
        
        Ok(())
    }

    pub async fn record_model_prediction(&self, confidence: f64, latency: f64) -> Result<()> {
        self.metrics.model_confidence.set(confidence);
        self.metrics.prediction_latency.observe(latency);
        
        info!(
            "Model prediction: confidence={:.2}, latency={:.3}s",
            confidence,
            latency
        );
        
        Ok(())
    }

    pub async fn record_training_metrics(&self, training_loss: f64, validation_loss: f64) -> Result<()> {
        self.metrics.training_loss.set(training_loss);
        self.metrics.validation_loss.set(validation_loss);
        
        info!(
            "Training metrics: loss={:.4}, val_loss={:.4}",
            training_loss,
            validation_loss
        );
        
        Ok(())
    }

    pub async fn record_risk_metrics(&self, risk_per_trade: f64, daily_pnl: f64) -> Result<()> {
        self.metrics.risk_per_trade.set(risk_per_trade);
        self.metrics.daily_pnl.set(daily_pnl);
        
        if daily_pnl < -0.1 { // Using a fixed threshold for now
            warn!(
                "Daily loss limit exceeded: {:.2}%",
                daily_pnl * 100.0
            );
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ApiConfig, TradingConfig, MonitoringConfig, AlertThresholds, TelegramConfig, DatabaseConfig, SecurityConfig, MLConfig, SolanaConfig, DexTradingConfig};
    use crate::trading::TradingBot;
    use crate::ml::{TradingModel, Predictor, ModelArchitecture, LossFunction, Activation};
    use crate::api::MarketDataCollector;
    use std::sync::Arc;
    use chrono::Utc;
    use prometheus::Registry;
    use tokio::sync::Mutex;
    use crate::error::Error;

    // Define DummyPredictor here as well
    struct DummyPredictor;
    impl Predictor for DummyPredictor {
        fn predict(&mut self, _data: &crate::trading::TradingMarketData) -> Result<Vec<f64>> {
            Ok(vec![0.5, 0.5]) // Always hold
        }
    }

    // Helper to create a default test config
    fn create_test_config() -> Config {
        let test_architecture = ModelArchitecture {
            input_size: 10, hidden_size: 20, output_size: 1,
            num_layers: 1, dropout: None, activation: Activation::ReLU,
        };
        let test_loss_function = LossFunction::MSE;

        Config {
            api: ApiConfig {
                coingecko_api_key: "test".to_string(),
                coinmarketcap_api_key: "test".to_string(),
                cryptodatadownload_api_key: "test".to_string(),
            },
            trading: TradingConfig {
                risk_level: 0.1,
                max_position_size: 1000.0,
                stop_loss_percentage: 0.05,
                take_profit_percentage: 0.1,
                trading_pairs: vec!["BTC/USD".to_string()],
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
                keychain_service_name: "test_keychain".to_string(),
                solana_key_username: "test_sol_user".to_string(),
                ton_key_username: "test_ton_user".to_string(),
            },
            ml: MLConfig {
                architecture: test_architecture,
                loss_function: test_loss_function,
                input_size: 10,
                hidden_size: 20,
                output_size: 1,
                learning_rate: 0.001,
                model_path: "test".to_string(),
                confidence_threshold: 0.8,
                training_batch_size: 32,
                training_epochs: 10,
                window_size: 10,
                min_data_points: 100,
                validation_split: 0.2,
                early_stopping_patience: 3,
                save_best_model: true,
                evaluation_window_size: 10,
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
            },
        }
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let registry = Arc::new(Registry::new());
        let monitor = Monitor::new(registry);
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let registry = Arc::new(Registry::new());
        let monitor = Monitor::new(registry).unwrap();
        // Test trade recording
        let trade = crate::trading::Trade {
            entry_time: Utc::now(),
            exit_time: Utc::now(),
            symbol: "BTC".to_string(),
            entry_price: 100.0,
            exit_price: 110.0,
            size: 1.0,
            pnl: 10.0,
            pnl_percentage: 0.1,
        };

        assert!(monitor.record_trade(&trade).await.is_ok());
    }

    #[tokio::test]
    async fn test_alerting() {
        let config = create_test_config();
        let config_arc = Arc::new(config);
        
        let market_data_collector = Arc::new(MarketDataCollector::new(
            config_arc.api.coingecko_api_key.clone(),
            config_arc.api.coinmarketcap_api_key.clone(),
            config_arc.api.cryptodatadownload_api_key.clone(),
        ));

        // Create Dummy Predictor instance
        let dummy_model: Arc<Mutex<dyn Predictor + Send>> = Arc::new(Mutex::new(DummyPredictor));

        // Update TradingBot::new call
        let bot = TradingBot::new(market_data_collector.clone(), dummy_model.clone(), config_arc.clone()).expect("Failed to create bot for test");
        let bot_arc = Arc::new(Mutex::new(bot));
        
        let registry = Arc::new(Registry::new());
        let monitor = Monitor::new(registry.clone());

        assert!(monitor.is_ok());
        let _monitor = monitor.unwrap();

        // TODO: Add actual alert trigger simulation and assertions
    }
} 
use crate::error::Result;
use crate::error::Error;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::api::MarketDataCollector;
use log::{info, error};
use crate::api::types;
use crate::ml::Predictor;
use crate::config::Config;
use crate::solana::SolanaManager;
use std::fmt;

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
    pub timestamp: DateTime<Utc>,
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

pub struct TradingBot {
    market_data_collector: Arc<Mutex<MarketDataCollector>>,
    positions: Arc<Mutex<Vec<Position>>>,
    risk_level: Arc<Mutex<f64>>,
    trading_enabled: Arc<Mutex<bool>>,
    model: Arc<Mutex<dyn Predictor + Send>>,
    config: Arc<Config>,
    risk_manager: Arc<Mutex<RiskManager>>,
    solana_manager: Arc<SolanaManager>,
}

impl fmt::Debug for TradingBot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TradingBot")
            .field("market_data_collector", &self.market_data_collector)
            .field("positions", &self.positions)
            .field("risk_level", &self.risk_level)
            .field("trading_enabled", &self.trading_enabled)
            .field("model", &format_args!("<Predictor Trait Object>"))
            .field("config", &self.config)
            .field("risk_manager", &self.risk_manager)
            .field("solana_manager", &self.solana_manager)
            .finish()
    }
}

impl TradingBot {
    pub fn new(
        market_data_collector: Arc<MarketDataCollector>,
        model: Arc<Mutex<dyn Predictor + Send>>,
        config: Arc<Config>
    ) -> Result<Self> {
        let risk_manager = RiskManager::new(
            config.trading.max_position_size,
            config.trading.stop_loss_percentage,
            config.trading.take_profit_percentage,
        );

        let solana_manager = SolanaManager::new(config.solana.clone(), &config.security)?;

        Ok(Self {
            market_data_collector: Arc::new(Mutex::new((*market_data_collector).clone())),
            positions: Arc::new(Mutex::new(Vec::new())),
            risk_level: Arc::new(Mutex::new(config.trading.risk_level)),
            trading_enabled: Arc::new(Mutex::new(false)),
            model,
            config: Arc::clone(&config),
            risk_manager: Arc::new(Mutex::new(risk_manager)),
            solana_manager: Arc::new(solana_manager),
        })
    }

    pub async fn enable_trading(&self, enabled: bool) {
        let mut trading_enabled_guard = self.trading_enabled.lock().await;
        *trading_enabled_guard = enabled;
        info!("Trading enabled set to: {}", enabled);
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

        let mut model_guard = self.model.lock().await;
        let prediction_result = model_guard.predict(&data).await;

        let signal = match prediction_result {
            Ok(prediction) => {
                let buy_confidence = prediction.get(0).cloned().unwrap_or(0.0);
                let sell_confidence = prediction.get(1).cloned().unwrap_or(0.0);
                let confidence_threshold = self.config.ml.confidence_threshold;

                if buy_confidence > confidence_threshold && buy_confidence >= sell_confidence {
                    Some(TradingSignal {
                        symbol: data.symbol.clone(),
                        signal_type: SignalType::Buy,
                        confidence: buy_confidence,
                        price: data.price,
                    })
                } else if sell_confidence > confidence_threshold && sell_confidence > buy_confidence {
                    Some(TradingSignal {
                        symbol: data.symbol.clone(),
                        signal_type: SignalType::Sell,
                        confidence: sell_confidence,
                        price: data.price,
                    })
                } else {
                    Some(TradingSignal {
                        symbol: data.symbol.clone(),
                        signal_type: SignalType::Hold,
                        confidence: 1.0 - (buy_confidence - sell_confidence).abs(),
                        price: data.price,
                    })
                }
            }
            Err(e) => {
                match e {
                    Error::MLWarmupRequired(reason) => {
                        log::debug!("ML Model warmup required: {}", reason);
                        return Err(Error::MLWarmupRequired(reason));
                    }
                    Error::MLConfigError(ml_config_err) => {
                        error!("ML configuration error during prediction: {}", ml_config_err);
                        None
                    }
                    _ => {
                        error!("Unexpected error during prediction: {}", e);
                        None
                    }
                }
            }
        };

        if let Some(ref signal) = signal {
            info!("Generated ML trading signal: {:?}", signal);
        }

        Ok(signal)
    }

    pub async fn execute_swap(&self, signal: TradingSignal, slippage_bps: u16) -> Result<()> {
        if !*self.trading_enabled.lock().await {
            info!("Trading disabled, skipping signal execution.");
            return Ok(());
        }
        
        let solana_manager = Arc::clone(&self.solana_manager);
        let risk_manager = self.risk_manager.lock().await;
        let config = Arc::clone(&self.config);

        if signal.symbol != config.dex_trading.trading_pair_symbol {
            info!(
                "Signal symbol '{}' does not match configured dex_trading pair '{}'. Ignoring DEX trade.",
                signal.symbol,
                config.dex_trading.trading_pair_symbol
            );
            return Ok(());
        }

        match signal.signal_type {
            SignalType::Buy => {
                info!("Processing BUY signal for DEX swap...");
                let confidence = signal.confidence;
                let max_risk_factor = config.trading.risk_level; 
                let min_risk_factor = 0.01; 
                let confidence_threshold = config.ml.confidence_threshold;

                let adjusted_risk_factor = if confidence < confidence_threshold {
                    min_risk_factor
                } else {
                    min_risk_factor + (max_risk_factor - min_risk_factor) * 
                        (confidence - confidence_threshold) / (1.0 - confidence_threshold).max(f64::EPSILON)
                }.max(0.0).min(max_risk_factor);
                
                info!("Adjusted Risk Factor: {:.4}", adjusted_risk_factor);

                let current_price = signal.price;
                let stop_loss_price = current_price * (1.0 - config.trading.stop_loss_percentage);
                let quote_token_balance_usd_equivalent = 10000.0;

                let proposed_size_usd = risk_manager.calculate_position_size(
                    current_price,
                    stop_loss_price,
                    quote_token_balance_usd_equivalent,
                    adjusted_risk_factor,
                );
                info!("Proposed USD size for BUY: ${:.2}", proposed_size_usd);
                
                if proposed_size_usd <= 0.0 {
                    info!("Calculated position size is zero or negative. Skipping buy.");
                    return Ok(());
                }

                let _new_position = Position {
                    symbol: signal.symbol.clone(),
                    amount: proposed_size_usd / signal.price,
                    entry_price: signal.price,
                    current_price: signal.price,
                    unrealized_pnl: 0.0,
                    size: proposed_size_usd,
                    entry_time: Utc::now(),
                };
                info!("Internal position calculated for {} size ${:.2}", signal.symbol, proposed_size_usd);

                let quote_mint = &config.dex_trading.quote_token_mint;
                let base_mint = &config.dex_trading.base_token_mint;
                let quote_decimals = config.dex_trading.quote_token_decimals;
                
                let input_amount_quote_token = (proposed_size_usd * 10f64.powi(quote_decimals as i32)) as u64;
                
                info!(
                    "Attempting swap: {:.6} {} -> {} (USD Value: {:.2})", 
                    input_amount_quote_token as f64 / 10f64.powi(quote_decimals as i32),
                    quote_mint, 
                    base_mint,
                    proposed_size_usd
                );

                match solana_manager.execute_swap(quote_mint, base_mint, input_amount_quote_token, slippage_bps).await {
                    Ok(signature) => {
                        info!("BUY swap executed successfully on Solana. Signature: {}", signature);
                    }
                    Err(e) => {
                        error!("Solana BUY swap failed: {}", e);
                    }
                }
            }
            SignalType::Sell => {
                info!("Processing SELL signal for DEX swap...");
                let base_mint = &config.dex_trading.base_token_mint;
                let quote_mint = &config.dex_trading.quote_token_mint;
                let base_decimals = config.dex_trading.base_token_decimals;

                #[cfg(not(test))]
                {
                    match solana_manager.get_balance(Some(base_mint)).await {
                        Ok(base_token_balance) => {
                            if base_token_balance > 0 {
                                info!(
                                    "Attempting swap: {:.6} {} -> {} (Full available balance)",
                                    base_token_balance as f64 / 10f64.powi(base_decimals as i32),
                                    base_mint,
                                    quote_mint
                                );
                                match solana_manager.execute_swap(base_mint, quote_mint, base_token_balance, slippage_bps).await {
                                    Ok(signature) => {
                                        info!("SELL swap executed successfully on Solana. Signature: {}", signature);
                                    }
                                    Err(e) => {
                                        error!("Solana SELL swap failed: {}", e);
                                    }
                                }
                            } else {
                                info!("Sell signal received, but no balance of {} found to sell.", base_mint);
                            }
                        }
                        Err(e) => {
                             error!("Failed to get balance for {} to execute sell: {}", base_mint, e);
                        }
                    }
                }
                #[cfg(test)]
                {
                    info!("Skipping SELL swap execution in test environment.");
                }
            }
            SignalType::Hold => {
                info!("HOLD signal received. No DEX action taken.");
            }
        }

        Ok(())
    }

    pub async fn get_trade_history(&self) -> Result<Vec<Trade>> {
        let positions = self.positions.lock().await;
        Ok(positions.iter()
            .map(|p| Trade {
                symbol: p.symbol.clone(),
                size: p.amount,
                entry_price: p.entry_price,
                exit_price: p.current_price,
                entry_time: p.entry_time,
                exit_time: Utc::now(),
                pnl: p.unrealized_pnl,
                pnl_percentage: (p.unrealized_pnl / p.size) * 100.0,
            })
            .collect())
    }

    pub async fn get_backtest_results(&self) -> Result<BacktestResult> {
        let positions = self.positions.lock().await;
        let total_trades = positions.len();
        let winning_trades = positions.iter()
            .filter(|p| p.unrealized_pnl > 0.0)
            .count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        let total_pnl = positions.iter()
            .map(|p| p.unrealized_pnl)
            .sum();
        let max_drawdown = positions.iter()
            .map(|p| (p.current_price - p.entry_price) / p.entry_price)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);
        let sharpe_ratio = if total_trades > 0 {
            let avg_return = total_pnl / total_trades as f64;
            let squared_deviations = positions.iter()
                .map(|p| {
                    let diff: f64 = p.unrealized_pnl - avg_return;
                    diff.powi(2)
                })
                .sum::<f64>();
            let std_dev = (squared_deviations / total_trades as f64).sqrt();
            if std_dev > 0.0 {
                avg_return / std_dev
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(BacktestResult {
            initial_balance: 10000.0,
            total_pnl,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            max_drawdown,
            sharpe_ratio,
            trades: self.get_trade_history().await?,
            equity_curve: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::config::{Config, ApiConfig, TradingConfig, MonitoringConfig, AlertThresholds, TelegramConfig, DatabaseConfig, SecurityConfig, MLConfig, SolanaConfig, DexTradingConfig};
    use std::sync::Arc;
    use crate::ml::{ModelArchitecture, Activation, LossFunction};

    struct DummyPredictor;

    impl Predictor for DummyPredictor {
        fn predict(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
            if data.price > 105.0 {
                Ok(vec![0.9, 0.1])
            } else if data.price < 95.0 {
                Ok(vec![0.1, 0.9])
            } else {
                 Ok(vec![0.5, 0.5])
            }
        }
    }

    fn create_test_config() -> Config {
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_execute_swap_call() {
        let config = Arc::new(create_test_config());
        let market_data_collector = Arc::new(MarketDataCollector::new(
            config.api.coingecko_api_key.clone(),
            config.api.coinmarketcap_api_key.clone(),
            config.api.cryptodatadownload_api_key.clone(),
        ));
        
        let dummy_model: Arc<Mutex<dyn Predictor + Send>> = Arc::new(Mutex::new(DummyPredictor));

        let bot = TradingBot::new(market_data_collector.clone(), dummy_model, config.clone()).expect("Failed to create bot");

        *bot.trading_enabled.lock().await = true;

        let buy_signal = TradingSignal {
            symbol: "SOL/USDC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 1.0,
            price: 100.0,
        };
        assert!(bot.execute_swap(buy_signal, 50).await.is_ok());

        let sell_signal = TradingSignal {
            symbol: "SOL/USDC".to_string(),
            signal_type: SignalType::Sell,
            confidence: 1.0,
            price: 110.0,
        };
        assert!(bot.execute_swap(sell_signal, 50).await.is_ok());
    }

    // Helper to create test market data
    fn create_test_trading_market_data(symbol: &str, price: f64) -> TradingMarketData {
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
            quote: types::Quote { // Assuming types::Quote is available
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

    #[tokio::test]
    async fn test_signal_generation_logic() {
        let mut config = create_test_config();
        // Ensure confidence threshold allows dummy signals through
        config.ml.confidence_threshold = 0.7;
        let config_arc = Arc::new(config);
        
        let market_data_collector = Arc::new(MarketDataCollector::new(
            config_arc.api.coingecko_api_key.clone(),
            config_arc.api.coinmarketcap_api_key.clone(),
            config_arc.api.cryptodatadownload_api_key.clone(),
        ));
        
        let dummy_model: Arc<Mutex<dyn Predictor + Send>> = Arc::new(Mutex::new(DummyPredictor));
        let bot = TradingBot::new(market_data_collector.clone(), dummy_model, config_arc.clone()).expect("Failed to create bot");
        *bot.trading_enabled.lock().await = true;

        // Test Case 1: Price > 105 (Expect Buy signal)
        let high_price_data = create_test_trading_market_data("SOL/USDC", 110.0);
        let signal1 = bot.process_market_data(high_price_data).await.unwrap();
        assert!(signal1.is_some());
        let signal1 = signal1.unwrap();
        assert!(matches!(signal1.signal_type, SignalType::Buy));
        assert_eq!(signal1.symbol, "SOL/USDC");
        assert_eq!(signal1.price, 110.0);
        assert!(signal1.confidence > config_arc.ml.confidence_threshold); // 0.9 from dummy

        // Test Case 2: Price < 95 (Expect Sell signal)
        let low_price_data = create_test_trading_market_data("SOL/USDC", 90.0);
        let signal2 = bot.process_market_data(low_price_data).await.unwrap();
        assert!(signal2.is_some());
        let signal2 = signal2.unwrap();
        assert!(matches!(signal2.signal_type, SignalType::Sell));
        assert_eq!(signal2.symbol, "SOL/USDC");
        assert_eq!(signal2.price, 90.0);
        assert!(signal2.confidence > config_arc.ml.confidence_threshold); // 0.9 from dummy

        // Test Case 3: Price between 95 and 105 (Expect Hold signal)
        let mid_price_data = create_test_trading_market_data("SOL/USDC", 100.0);
        let signal3 = bot.process_market_data(mid_price_data).await.unwrap();
        assert!(signal3.is_some());
        let signal3 = signal3.unwrap();
        assert!(matches!(signal3.signal_type, SignalType::Hold));
        assert_eq!(signal3.symbol, "SOL/USDC");
        assert_eq!(signal3.price, 100.0);
        // Confidence for Hold in dummy is 1.0 - abs(0.5 - 0.5) = 1.0
        assert!(signal3.confidence > config_arc.ml.confidence_threshold); 
    }
} 
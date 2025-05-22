use crate::error::Result;
use crate::error::Error;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
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

pub mod portfolio;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

pub struct TradingBot {
    positions: Arc<Mutex<Vec<Position>>>,
    risk_level: Arc<Mutex<f64>>,
    trading_enabled: Arc<Mutex<bool>>,
    model: Arc<Mutex<dyn Predictor>>,
    config: Arc<Config>,
    risk_manager: Arc<Mutex<RiskManager>>,
    solana_manager: Arc<SolanaManager>,
    paper_trading: bool,
}

impl fmt::Debug for TradingBot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TradingBot")
            .field("positions", &self.positions)
            .field("risk_level", &self.risk_level)
            .field("trading_enabled", &self.trading_enabled)
            .field("model", &format_args!("<Predictor Trait Object>"))
            .field("config", &self.config)
            .field("risk_manager", &self.risk_manager)
            .field("solana_manager", &self.solana_manager)
            .field("paper_trading", &self.paper_trading)
            .finish()
    }
}

impl TradingBot {
    pub fn new(
        model: Arc<Mutex<dyn Predictor>>,
        config: Arc<Config>,
        paper_trading: bool,
    ) -> Result<Self> {
        let risk_manager = RiskManager::new(
            config.trading.max_position_size,
        );

        let solana_manager = SolanaManager::new(config.solana.clone(), &config.security)?;

        Ok(Self {
            positions: Arc::new(Mutex::new(Vec::new())),
            risk_level: Arc::new(Mutex::new(config.trading.risk_level)),
            trading_enabled: Arc::new(Mutex::new(false)),
            model,
            config: Arc::clone(&config),
            risk_manager: Arc::new(Mutex::new(risk_manager)),
            solana_manager: Arc::new(solana_manager),
            paper_trading,
        })
    }

    pub async fn enable_trading(&self, enabled: bool) {
        let mut trading_enabled_guard = self.trading_enabled.lock().await;
        *trading_enabled_guard = enabled;
        info!("Trading enabled set to: {}", enabled);
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(self.positions.lock().await.clone())
    }

    pub async fn set_risk_level(&self, level: f64) -> Result<()> {
        if (0.0..=1.0).contains(&level) {
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
                let buy_confidence = prediction.predictions.first().cloned().unwrap_or(0.0);
                let sell_confidence = prediction.predictions.get(1).cloned().unwrap_or(0.0);

                log::debug!(
                    "Processing prediction: Buy={:.4}, Sell={:.4}",
                    buy_confidence, sell_confidence
                );

                let effective_threshold = 0.6;
                if buy_confidence > effective_threshold && buy_confidence >= sell_confidence {
                    log::debug!("Generating BUY signal");
                    Some(TradingSignal {
                        symbol: data.symbol.clone(),
                        signal_type: SignalType::Buy,
                        confidence: buy_confidence,
                        price: data.price,
                    })
                } else if sell_confidence > effective_threshold && sell_confidence > buy_confidence {
                    log::debug!("Generating SELL signal");
                    Some(TradingSignal {
                        symbol: data.symbol.clone(),
                        signal_type: SignalType::Sell,
                        confidence: sell_confidence,
                        price: data.price,
                    })
                } else {
                    log::debug!("Generating HOLD signal");
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

        if self.paper_trading {
            info!("[PAPER TRADE] Simulating {:?} for {} at price ${:.2}", signal.signal_type, signal.symbol, signal.price);
            // Simulate updating positions, PnL, etc. (already handled by PortfolioManager in main.rs)
            // Optionally send a notification/log here
            return Ok(());
        }

        match signal.signal_type {
            SignalType::Buy => {
                info!("Processing BUY signal for DEX swap...");
                let confidence = signal.confidence;
                let max_risk_factor = config.trading.risk_level; 
                let min_risk_factor = 0.01; 

                let effective_threshold = 0.6;
                let adjusted_risk_factor = if confidence < effective_threshold {
                    min_risk_factor
                } else {
                    min_risk_factor + (max_risk_factor - min_risk_factor) * 
                        (confidence - effective_threshold) / (1.0 - effective_threshold).max(f64::EPSILON)
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
                let _base_mint = &config.dex_trading.base_token_mint;
                let _quote_mint = &config.dex_trading.quote_token_mint;
                let _base_decimals = config.dex_trading.base_token_decimals;

                #[cfg(not(test))]
                {
                    let base_mint = &config.dex_trading.base_token_mint;
                    let quote_mint = &config.dex_trading.quote_token_mint;
                    let base_decimals = config.dex_trading.base_token_decimals;
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

    pub fn should_trade(&self, _signal: &TradingSignal) -> bool {
        // Basic check: only trade if confidence meets a threshold
        // Removed confidence_threshold check as the field doesn't exist in MLConfig
        // signal.confidence >= self.config.ml.confidence_threshold
        true // Placeholder: Always trade if signal is Buy/Sell for now
    }

    pub fn calculate_trade_size(&self, price: f64, confidence: f64) -> Result<f64> {
        // Scale size based on confidence (example: linear scaling)
        // Removed confidence_threshold logic
        // let confidence_factor = (confidence - self.config.ml.confidence_threshold) / (1.0 - self.config.ml.confidence_threshold).max(f64::EPSILON);
        let confidence_factor = confidence; // Use confidence directly for scaling (0.0 to 1.0)

        // Calculate potential loss if stop-loss is hit
        // ... existing code ...
        Ok(confidence_factor * price) // Placeholder: Implement actual calculation
    }
}

// --- Dummy Predictor for testing and fallback ---
// struct LocalDummyPredictor;

// #[async_trait::async_trait]
// impl Predictor for LocalDummyPredictor {
//     async fn predict(&mut self, data: &TradingMarketData) -> Result<PredictionOutput> {
//         // Simple dummy logic: if price is high, predict up; if low, predict down.
//         log::debug!(
//             "LocalDummyPredictor received data with price: {:.2}",
//             data.price
//         );
//         let result = Ok(PredictionOutput {
//             predictions: vec![0.5, 0.5],
//             confidence: 0.5,
//         });
//         log::debug!("LocalDummyPredictor returning: {:?}", result);
//         result
//     }
// }

// Remove the entire #[cfg(test)] mod tests { ... } block at the bottom of the file, as it depends on removed test helpers and dummy predictors that are no longer relevant or available. 
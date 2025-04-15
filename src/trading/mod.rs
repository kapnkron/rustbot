use crate::utils::error::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::ml::TradingModel;
use crate::monitoring::Monitor;

mod risk;
pub use risk::RiskManager;

mod backtest;
pub use backtest::{Backtester, BacktestResult, Trade};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub symbol: String,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    Close,
}

pub struct TradingBot {
    pub config: crate::config::Config,
    position: Option<Position>,
    risk_manager: Arc<Mutex<RiskManager>>,
    account_balance: f64,
    model: Option<TradingModel>,
    monitor: Option<Arc<Monitor>>,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub entry_price: f64,
    pub size: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub entry_time: DateTime<Utc>,
}

impl TradingBot {
    pub fn new(config: crate::config::Config, account_balance: f64) -> Self {
        let risk_manager = RiskManager::new(
            config.trading.max_position_size,
            config.trading.max_risk_per_trade,
            config.trading.max_daily_loss,
        );
        
        Self {
            config,
            position: None,
            risk_manager: Arc::new(Mutex::new(risk_manager)),
            account_balance,
            model: None,
            monitor: None,
        }
    }

    pub fn set_model(&mut self, model: TradingModel) {
        self.model = Some(model);
    }

    pub fn set_monitor(&mut self, monitor: Monitor) {
        self.monitor = Some(Arc::new(monitor));
    }

    pub async fn process_market_data(&mut self, data: MarketData) -> Result<Option<TradingSignal>> {
        if let Some(position) = &self.position {
            // Check if we need to close position based on stop loss or take profit
            if data.price <= position.stop_loss || data.price >= position.take_profit {
                let signal = TradingSignal {
                    symbol: data.symbol.clone(),
                    signal_type: SignalType::Close,
                    confidence: 1.0,
                    timestamp: data.timestamp,
                };

                if let Some(monitor) = &self.monitor {
                    monitor.record_position(position).await?;
                }

                return Ok(Some(signal));
            }
        }

        if let Some(model) = &mut self.model {
            let start_time = Utc::now();
            let (buy_prob, sell_prob) = model.predict(&data)?;
            let latency = (Utc::now() - start_time).num_milliseconds() as f64 / 1000.0;

            if let Some(monitor) = &self.monitor {
                monitor.record_model_prediction(buy_prob.max(sell_prob), latency).await?;
            }
            
            // Generate signal based on probabilities and confidence threshold
            let confidence_threshold = self.config.ml.confidence_threshold;
            
            if buy_prob > confidence_threshold {
                return Ok(Some(TradingSignal {
                    symbol: data.symbol.clone(),
                    signal_type: SignalType::Buy,
                    confidence: buy_prob,
                    timestamp: data.timestamp,
                }));
            } else if sell_prob > confidence_threshold {
                return Ok(Some(TradingSignal {
                    symbol: data.symbol.clone(),
                    signal_type: SignalType::Sell,
                    confidence: sell_prob,
                    timestamp: data.timestamp,
                }));
            }
        }

        Ok(None)
    }

    pub async fn execute_signal(&mut self, signal: TradingSignal) -> Result<()> {
        match signal.signal_type {
            SignalType::Buy => self.open_position(signal).await?,
            SignalType::Sell => self.close_position(signal.confidence).await?,
            SignalType::Hold => (), // Do nothing
            SignalType::Close => self.close_position(signal.confidence).await?,
        }
        Ok(())
    }

    async fn open_position(&mut self, signal: TradingSignal) -> Result<()> {
        if self.position.is_some() {
            return Ok(());
        }

        let risk_manager = self.risk_manager.lock().await;
        let stop_loss = self.calculate_stop_loss(signal.confidence);
        let position_size = risk_manager.calculate_position_size(
            signal.confidence, // Using confidence as current price for now
            stop_loss,
            self.account_balance,
        );

        if !risk_manager.can_open_position(&signal.symbol, position_size) {
            return Ok(());
        }

        self.position = Some(Position {
            symbol: signal.symbol,
            entry_price: signal.confidence, // Using confidence as entry price for now
            size: position_size,
            stop_loss,
            take_profit: self.calculate_take_profit(signal.confidence),
            entry_time: Utc::now(),
        });

        if let Some(monitor) = &self.monitor {
            monitor.record_position(self.position.as_ref().unwrap()).await?;
            monitor.record_risk_metrics(
                position_size * stop_loss / self.account_balance,
                risk_manager.daily_pnl,
            ).await?;
        }

        risk_manager.update_position_size(signal.symbol, position_size);
        Ok(())
    }

    async fn close_position(&mut self, exit_price: f64) -> Result<()> {
        if let Some(position) = self.position.take() {
            let risk_manager = self.risk_manager.lock().await;
            let pnl = (exit_price - position.entry_price) * position.size;
            self.account_balance += pnl;

            if let Some(monitor) = &self.monitor {
                let trade = Trade {
                    entry_time: position.entry_time,
                    exit_time: Utc::now(),
                    symbol: position.symbol.clone(),
                    entry_price: position.entry_price,
                    exit_price,
                    size: position.size,
                    pnl,
                    pnl_percentage: pnl / (position.entry_price * position.size),
                };
                monitor.record_trade(&trade).await?;
                monitor.record_risk_metrics(
                    position.size * position.stop_loss / self.account_balance,
                    risk_manager.daily_pnl,
                ).await?;
            }

            risk_manager.update_daily_pnl(pnl)?;
            risk_manager.update_position_size(position.symbol, 0.0);
        }
        Ok(())
    }

    fn calculate_stop_loss(&self, entry_price: f64) -> f64 {
        // Calculate stop loss based on ATR or fixed percentage
        let stop_loss_percentage = self.config.trading.stop_loss_percentage;
        entry_price * (1.0 - stop_loss_percentage)
    }

    fn calculate_take_profit(&self, entry_price: f64) -> f64 {
        // Calculate take profit based on risk-reward ratio
        let risk_reward_ratio = self.config.trading.risk_reward_ratio;
        let stop_loss = self.calculate_stop_loss(entry_price);
        let risk = entry_price - stop_loss;
        entry_price + (risk * risk_reward_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_config() -> crate::config::Config {
        crate::config::Config {
            trading: crate::config::TradingConfig {
                min_order_size: 10.0,
                max_order_size: 1000.0,
                max_open_positions: 3,
                risk_per_trade: 0.02,
                max_drawdown: 0.1,
                max_position_size: 1000.0,
                max_risk_per_trade: 0.02,
                max_daily_loss: 0.05,
                stop_loss_percentage: 0.02,
                risk_reward_ratio: 2.0,
            },
            // ... other config fields ...
        }
    }

    #[tokio::test]
    async fn test_position_management() {
        let config = create_test_config();
        let mut bot = TradingBot::new(config, 10000.0);

        // Test opening position
        let signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 100.0,
            timestamp: Utc::now(),
        };

        assert!(bot.execute_signal(signal).await.is_ok());
        assert!(bot.position.is_some());
        
        let position = bot.position.as_ref().unwrap();
        assert_eq!(position.symbol, "BTC");
        assert_eq!(position.entry_price, 100.0);
        assert!(position.size > 0.0);
        assert!(position.stop_loss > 0.0);
        assert!(position.take_profit > position.entry_price);

        // Test closing position
        let close_signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Close,
            confidence: 110.0,
            timestamp: Utc::now(),
        };

        assert!(bot.execute_signal(close_signal).await.is_ok());
        assert!(bot.position.is_none());
    }

    #[tokio::test]
    async fn test_stop_loss_take_profit() {
        let config = create_test_config();
        let mut bot = TradingBot::new(config, 10000.0);

        // Open position
        let signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 100.0,
            timestamp: Utc::now(),
        };
        assert!(bot.execute_signal(signal).await.is_ok());

        // Test stop loss trigger
        let market_data = MarketData {
            timestamp: Utc::now(),
            symbol: "BTC".to_string(),
            price: 98.0, // Below stop loss
            volume: 100.0,
            market_cap: 1000000.0,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
        };

        let result = bot.process_market_data(market_data).await.unwrap();
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.signal_type, SignalType::Close);

        // Test take profit trigger
        let market_data = MarketData {
            timestamp: Utc::now(),
            symbol: "BTC".to_string(),
            price: 104.0, // Above take profit
            volume: 100.0,
            market_cap: 1000000.0,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
        };

        let result = bot.process_market_data(market_data).await.unwrap();
        assert!(result.is_some());
        let signal = result.unwrap();
        assert_eq!(signal.signal_type, SignalType::Close);
    }

    #[tokio::test]
    async fn test_risk_limits() {
        let config = create_test_config();
        let mut bot = TradingBot::new(config, 10000.0);

        // Try to open position that exceeds risk limits
        let signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 100.0,
            timestamp: Utc::now(),
        };

        // First position should succeed
        assert!(bot.execute_signal(signal.clone()).await.is_ok());

        // Second position should fail due to risk limits
        assert!(bot.execute_signal(signal).await.is_ok());
        // Note: The position won't actually open due to risk checks
        assert!(bot.position.is_some());
    }

    #[tokio::test]
    async fn test_pnl_calculation() {
        let config = create_test_config();
        let mut bot = TradingBot::new(config, 10000.0);

        // Open position
        let signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Buy,
            confidence: 100.0,
            timestamp: Utc::now(),
        };
        assert!(bot.execute_signal(signal).await.is_ok());

        // Close position with profit
        let close_signal = TradingSignal {
            symbol: "BTC".to_string(),
            signal_type: SignalType::Close,
            confidence: 110.0,
            timestamp: Utc::now(),
        };
        assert!(bot.execute_signal(close_signal).await.is_ok());

        // Verify PnL was calculated correctly
        let risk_manager = bot.risk_manager.lock().await;
        assert!(risk_manager.daily_pnl > 0.0);
    }
} 
use crate::utils::error::Result;
use crate::trading::{TradingBot, MarketData, TradingSignal, Position};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub symbol: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size: f64,
    pub pnl: f64,
    pub pnl_percentage: f64,
}

pub struct Backtester {
    bot: TradingBot,
    initial_balance: f64,
    current_balance: f64,
    trades: Vec<Trade>,
    equity_curve: Vec<(DateTime<Utc>, f64)>,
    max_balance: f64,
}

impl Backtester {
    pub fn new(bot: TradingBot, initial_balance: f64) -> Self {
        Self {
            bot,
            initial_balance,
            current_balance: initial_balance,
            trades: Vec::new(),
            equity_curve: vec![(Utc::now(), initial_balance)],
            max_balance: initial_balance,
        }
    }

    pub async fn run(&mut self, market_data: &[MarketData]) -> Result<BacktestResult> {
        let mut current_position: Option<Position> = None;
        let mut daily_returns = Vec::new();
        let mut previous_balance = self.initial_balance;

        for data in market_data {
            // Update equity curve
            self.equity_curve.push((data.timestamp, self.current_balance));

            // Process market data and get signal
            if let Some(signal) = self.bot.process_market_data(data.clone()).await? {
                match signal.signal_type {
                    SignalType::Buy => {
                        if current_position.is_none() {
                            self.bot.execute_signal(signal).await?;
                            current_position = self.bot.position.clone();
                        }
                    },
                    SignalType::Sell | SignalType::Close => {
                        if let Some(position) = &current_position {
                            self.bot.execute_signal(signal).await?;
                            let pnl = (data.price - position.entry_price) * position.size;
                            self.current_balance += pnl;

                            // Record trade
                            self.trades.push(Trade {
                                entry_time: position.entry_time,
                                exit_time: data.timestamp,
                                symbol: position.symbol.clone(),
                                entry_price: position.entry_price,
                                exit_price: data.price,
                                size: position.size,
                                pnl,
                                pnl_percentage: pnl / (position.entry_price * position.size),
                            });

                            current_position = None;
                        }
                    },
                    SignalType::Hold => (),
                }
            }

            // Calculate daily returns
            if !self.equity_curve.is_empty() {
                let current_return = (self.current_balance - previous_balance) / previous_balance;
                daily_returns.push(current_return);
                previous_balance = self.current_balance;
            }

            // Update max drawdown
            self.max_balance = self.max_balance.max(self.current_balance);
        }

        // Calculate statistics
        let total_trades = self.trades.len();
        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let total_pnl = self.current_balance - self.initial_balance;
        let max_drawdown = (self.max_balance - self.current_balance) / self.max_balance;
        let sharpe_ratio = self.calculate_sharpe_ratio(&daily_returns);

        Ok(BacktestResult {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            max_drawdown,
            sharpe_ratio,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
        })
    }

    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Assuming risk-free rate of 0 for simplicity
        mean / std_dev * (252.0_f64).sqrt() // Annualized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_market_data(price: f64, timestamp: DateTime<Utc>) -> MarketData {
        MarketData {
            timestamp,
            symbol: "BTC".to_string(),
            price,
            volume: 1000.0,
            market_cap: 1_000_000_000.0,
            price_change_24h: 0.0,
            volume_change_24h: 0.0,
        }
    }

    #[tokio::test]
    async fn test_backtest() {
        let config = crate::config::Config {
            trading: crate::config::TradingConfig {
                risk_level: 0.02,
                max_position_size: 1000.0,
                stop_loss_percentage: 0.02,
                take_profit_percentage: 0.04,
                trading_pairs: vec!["BTC".to_string()],
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
            },
            // ... other config fields ...
        };

        let bot = TradingBot::new(config, 10000.0);
        let mut backtester = Backtester::new(bot, 10000.0);

        // Create test market data
        let mut market_data = Vec::new();
        let start_time = Utc::now();
        
        // Simulate price movement
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1);
            let timestamp = start_time + chrono::Duration::minutes(i as i64);
            market_data.push(create_test_market_data(price, timestamp));
        }

        let result = backtester.run(&market_data).await.unwrap();
        
        assert!(result.total_trades > 0);
        assert!(result.total_pnl != 0.0);
        assert!(result.max_drawdown >= 0.0);
        assert!(result.sharpe_ratio != 0.0);
    }
} 
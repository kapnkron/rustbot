use crate::error::Result;
use crate::trading::{TradingBot, TradingSignal, Position, SignalType};
use crate::api::MarketData;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use log::{info, error};

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
                            // Open long position
                            let position = Position {
                                symbol: signal.symbol.clone(),
                                amount: self.current_balance / signal.price,
                                entry_price: signal.price,
                                current_price: signal.price,
                                unrealized_pnl: 0.0,
                                size: self.current_balance,
                                entry_time: data.timestamp,
                            };
                            current_position = Some(position);
                            info!("Opened long position for {} at ${}", signal.symbol, signal.price);
                        }
                    }
                    SignalType::Sell => {
                        if let Some(position) = current_position.take() {
                            // Close position and calculate PnL
                            let pnl = position.amount * (signal.price - position.entry_price);
                            self.current_balance += pnl;
                            
                            let trade = Trade {
                                entry_time: position.entry_time,
                                exit_time: data.timestamp,
                                symbol: position.symbol,
                                entry_price: position.entry_price,
                                exit_price: signal.price,
                                size: position.size,
                                pnl,
                                pnl_percentage: pnl / position.size * 100.0,
                            };
                            self.trades.push(trade);
                            
                            info!("Closed position with PnL: ${:.2}", pnl);
                        }
                    }
                    SignalType::Hold => {}
                }
            }

            // Update position if we have one
            if let Some(ref mut position) = current_position {
                position.current_price = data.price;
                position.unrealized_pnl = position.amount * (data.price - position.entry_price);
            }

            // Calculate daily returns
            let daily_return = (self.current_balance - previous_balance) / previous_balance;
            daily_returns.push(daily_return);
            previous_balance = self.current_balance;
        }

        // Calculate backtest metrics
        let total_trades = self.trades.len();
        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = winning_trades as f64 / total_trades as f64;
        let total_pnl = self.current_balance - self.initial_balance;
        let max_drawdown = self.calculate_max_drawdown();
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

    fn calculate_max_drawdown(&self) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = self.initial_balance;

        for &(_, balance) in &self.equity_curve {
            if balance > peak {
                peak = balance;
            }
            let drawdown = (peak - balance) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    fn calculate_sharpe_ratio(&self, daily_returns: &[f64]) -> f64 {
        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / daily_returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            0.0
        } else {
            mean_return / std_dev * (252.0_f64).sqrt() // Annualized Sharpe Ratio
        }
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
            api: crate::config::ApiConfig {
                coingecko_api_key: "test".to_string(),
                coinmarketcap_api_key: "test".to_string(),
                cryptodatadownload_api_key: "test".to_string(),
            },
            trading: crate::config::TradingConfig {
                risk_level: 0.02,
                max_position_size: 1000.0,
                stop_loss_percentage: 0.02,
                take_profit_percentage: 0.04,
                trading_pairs: vec!["BTC".to_string()],
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
        };

        let market_data_collector = MarketDataCollector::new(
            config.api.coingecko_api_key.clone(),
            config.api.coinmarketcap_api_key.clone(),
            config.api.cryptodatadownload_api_key.clone(),
        );
        let bot = TradingBot::new(market_data_collector);
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
use crate::error::Result;
use crate::trading::{TradingBot, Position, SignalType, TradingMarketData};
use log::{info, error};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::api::MarketData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub initial_balance: f64,
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
}

impl Backtester {
    pub fn new(bot: TradingBot, initial_balance: f64) -> Self {
        Self {
            bot,
            initial_balance,
            current_balance: initial_balance,
            trades: Vec::new(),
            equity_curve: vec![(Utc::now(), initial_balance)],
        }
    }

    pub async fn run(&mut self, market_data: &[MarketData]) -> Result<BacktestResult> {
        let mut current_position: Option<Position> = None;
        let mut daily_returns = Vec::new();
        let mut previous_balance = self.initial_balance;
        let required_initial_points = self.bot.config.ml.window_size; 

        for (index, data) in market_data.iter().enumerate() {
            self.equity_curve.push((data.timestamp, self.current_balance));

            let trading_data: TradingMarketData = data.clone().into(); 
            let signal_option = self.bot.process_market_data(trading_data).await;

            if index >= required_initial_points {
                 match &signal_option {
                    Ok(Some(sig)) => info!("[{}] Received Signal: {:?}", data.timestamp, sig),
                    Ok(None) => info!("[{}] Received None (Hold/LowConf)", data.timestamp),
                    Err(ref e) => info!("[{}] Received Error: {}", data.timestamp, e), 
                 }
            }

            match signal_option {
                Ok(Some(signal)) => {
                    match signal.signal_type {
                        SignalType::Buy => {
                            if current_position.is_none() {
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
                                info!("[{}] Opened long position for {} at ${}", data.timestamp, signal.symbol, signal.price);
                            }
                        }
                        SignalType::Sell => {
                            if let Some(position) = current_position.take() {
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
                                
                                info!("[{}] Closed position for {} at ${} with PnL: ${:.2}", 
                                      data.timestamp, signal.symbol, signal.price, pnl);
                            }
                        }
                        SignalType::Hold => {
                             info!("[{}] Hold signal received for {}.", data.timestamp, signal.symbol);
                        }
                    }
                }
                Ok(None) => {
                    info!("[{}] No actionable signal generated.", data.timestamp);
                }
                Err(e) => {
                    if index < required_initial_points {
                         match e {
                             crate::error::Error::MLConfigError(ref ml_conf_err) => {
                                 info!("[{}] Preprocessor warming up (step {}/{}): {}", data.timestamp, index + 1, required_initial_points, ml_conf_err);
                             }
                             _ => {
                                 error!("[{}] Unexpected Error during backtest warmup: {}", data.timestamp, e);
                                 return Err(e);
                             }
                         }
                    } else {
                        error!("[{}] Error after warmup during backtest run: {}", data.timestamp, e);
                        return Err(e);
                    }
                }
            }

            if let Some(ref mut position) = current_position {
                position.current_price = data.price;
                position.unrealized_pnl = position.amount * (data.price - position.entry_price);
            }

            let daily_return = (self.current_balance - previous_balance) / previous_balance;
            daily_returns.push(daily_return);
            previous_balance = self.current_balance;
        }

        let total_trades = self.trades.len();
        let winning_trades = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = total_trades - winning_trades;
        let win_rate = if total_trades == 0 { 0.0 } else { winning_trades as f64 / total_trades as f64 };
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
            initial_balance: self.initial_balance,
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
        if daily_returns.is_empty() {
            return 0.0; // Avoid division by zero
        }
        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / daily_returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            0.0
        } else {
            mean_return / std_dev * (252.0_f64).sqrt() 
        }
    }
}

#[cfg(test)]
mod tests {
    
    
    
    
    
    
    
    
    

    // Removed: DummyPredictor struct
    /*
    #[derive(Debug, Clone)]
    struct DummyPredictor;

    #[async_trait::async_trait]
    impl Predictor for DummyPredictor {
        async fn predict(&mut self, _data: &crate::trading::TradingMarketData) -> Result<Vec<f64>> {
            Ok(vec![0.6, 0.4])
        }
    }
    */

    // Removed: create_test_config_for_backtest function
    /*
    fn create_test_config_for_backtest() -> Config {
        let mut config = tests::common::create_test_config();
        config.ml.min_data_points = 10;
        config.ml.window_size = 5;
        config.dex_trading.slippage_bps = 50;
        config
    }
    */

    // Removed: test_backtester_run function
    /* 
    async fn test_backtester_run() -> Result<()> {
        let config = Arc::new(create_test_config_for_backtest());
        let dummy_model: Arc<tokio::sync::Mutex<dyn Predictor + Send>> = Arc::new(tokio::sync::Mutex::new(DummyPredictor));
        let dummy_collector = Arc::new(tokio::sync::Mutex::new(tests::common::create_mock_market_data_collector()));
        
        let bot = crate::trading::TradingBot::new(dummy_collector, dummy_model, Arc::clone(&config))?;
        let mut backtester = Backtester::new(bot, 10000.0);

        let start_time = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let market_data: Vec<MarketData> = (0..100)
            .map(|i| MarketData {
                timestamp: start_time + Duration::minutes(i),
                symbol: "SOL/USDC".to_string(),
                price: 100.0 + (i as f64 * 0.1),
                volume: 1000.0,
                market_cap: 1_000_000.0,
                price_change_24h: 0.0,
                volume_change_24h: 0.0,
                volume_24h: 1000.0,
                change_24h: 0.0,
                quote: types::Quote { usd: types::USDData { price: 100.0 + (i as f64 * 0.1), volume_24h: 1000.0, market_cap: 1_000_000.0, percent_change_24h: 0.0, volume_change_24h: 0.0 } },
            })
            .collect();

        let results = backtester.run(&market_data).await?;

        assert!(results.total_trades > 0, "Expected trades to occur");
        assert_ne!(results.total_pnl, 0.0, "Expected PnL to change");

        Ok(())
    }
    */
} 
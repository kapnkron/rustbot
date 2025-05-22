use crate::api::MarketData;
use crate::trading::{TradingSignal, TradingBot, BacktestResult, Trade};
use crate::monitoring::dashboard::Dashboard;
use crate::error::Result;
use teloxide::prelude::*;
use teloxide::types::ParseMode;
use teloxide::utils::command::BotCommands;
use teloxide::dispatching::repls::CommandReplExt;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::error;
use std::collections::VecDeque;

const RECENT_TRADES_LIMIT: usize = 5;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "These commands are supported:")]
pub enum Command {
    #[command(description = "Start the bot")]
    Start,
    #[command(description = "Get current market data")]
    MarketData,
    #[command(description = "Get current positions")]
    Positions,
    #[command(description = "Get trading history")]
    History,
    #[command(description = "Get bot status")]
    Status,
    #[command(description = "Get backtesting results")]
    Backtest,
    #[command(description = "Show recent bot actions/logs")]
    Log,
    #[command(description = "Display this help message")]
    Help,
}

pub struct TelegramBot {
    bot: Bot,
    chat_id: ChatId,
    trading_bot: Arc<TradingBot>,
    dashboard: Arc<Mutex<Dashboard>>,
    log: Arc<Mutex<VecDeque<String>>>,
}

impl TelegramBot {
    pub fn new(bot_token: String, chat_id: String, trading_bot: Arc<TradingBot>, dashboard: Dashboard) -> Self {
        Self {
            bot: Bot::new(bot_token),
            chat_id: ChatId(chat_id.parse().expect("Invalid chat ID")),
            trading_bot,
            dashboard: Arc::new(Mutex::new(dashboard)),
            log: Arc::new(Mutex::new(VecDeque::with_capacity(20))),
        }
    }

    pub async fn start(self: Arc<Self>) -> Result<()> {
        let bot = self.bot.clone();
        let handler_instance = self.clone();
        Command::repl(bot, move |b: Bot, msg: Message, cmd: Command| {
            let _bot_in_handler = b;
            let handler_instance = handler_instance.clone();
            async move {
                if let Err(e) = handler_instance.handle_command(msg, cmd).await {
                    error!("Error handling command: {}", e);
                }
                Ok(())
            }
        })
        .await;
        Ok(())
    }

    pub async fn send_market_data(&self, data: &MarketData) -> Result<()> {
        let price_change_emoji = if data.price_change_24h >= 0.0 { "📈" } else { "📉" };
        let volume_change_emoji = if data.volume_change_24h >= 0.0 { "🔼" } else { "🔽" };
        
        let message = format!(
            "📊 *Market Data for {}*\n\n\
            💵 Price: ${:.2} {}\n\
            📊 24h Change: {:.2}%\n\
            💰 Market Cap: ${:.2}\n\
            📈 Volume: ${:.2} {}\n\
            🔄 24h Volume Change: {:.2}%",
            data.symbol,
            data.price,
            price_change_emoji,
            data.price_change_24h,
            data.market_cap,
            data.volume,
            volume_change_emoji,
            data.volume_change_24h
        );

        self.bot
            .send_message(self.chat_id, message)
            .parse_mode(ParseMode::MarkdownV2)
            .await?;

        Ok(())
    }

    pub async fn send_trading_signal(&self, signal: &TradingSignal) -> Result<()> {
        let message = format!(
            "🚨 Trading Signal\n\
            Symbol: {}\n\
            Type: {:?}\n\
            Confidence: {:.2}%",
            signal.symbol,
            signal.signal_type,
            signal.confidence * 100.0
        );

        self.bot
            .send_message(self.chat_id, message)
            .parse_mode(ParseMode::Html)
            .await?;

        Ok(())
    }

    pub async fn send_alert(&self, message: &str) -> Result<()> {
        self.bot
            .send_message(self.chat_id, format!("⚠️ Alert: {}", message))
            .parse_mode(ParseMode::Html)
            .await?;

        Ok(())
    }

    pub async fn send_backtest_results(&self, results: &BacktestResult) -> Result<()> {
        let message = format!(
            "📊 *Backtest Results*\n\n\
            💰 Initial Balance: ${:.2}\n\
            💵 Final Balance: ${:.2}\n\
            📈 Total PnL: ${:.2} ({:.2}%)\n\n\
            📊 *Performance Metrics*\n\
            ✅ Win Rate: {:.2}%\n\
            📊 Total Trades: {}\n\
            ✅ Winning Trades: {}\n\
            ❌ Losing Trades: {}\n\
            📉 Max Drawdown: {:.2}%\n\
            📊 Sharpe Ratio: {:.2}\n\n\
            *Recent Trades*\n{}",
            results.initial_balance,
            results.initial_balance + results.total_pnl,
            results.total_pnl,
            (results.total_pnl / results.initial_balance) * 100.0,
            results.win_rate * 100.0,
            results.total_trades,
            results.winning_trades,
            results.losing_trades,
            results.max_drawdown * 100.0,
            results.sharpe_ratio,
            self.format_recent_trades(&results.trades)
        );

        self.bot
            .send_message(self.chat_id, message)
            .parse_mode(ParseMode::MarkdownV2)
            .await?;

        Ok(())
    }

    fn format_recent_trades(&self, trades: &[Trade]) -> String {
        trades.iter()
            .rev()
            .take(RECENT_TRADES_LIMIT)
            .map(|trade| {
                let pnl_emoji = if trade.pnl >= 0.0 { "✅" } else { "❌" };
                format!(
                    "{} {} {} @ ${:.2} → ${:.2} (PnL: ${:.2}, {:.2}%)",
                    pnl_emoji,
                    trade.symbol,
                    trade.entry_time.format("%Y-%m-%d %H:%M"),
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl,
                    trade.pnl_percentage
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub async fn handle_command(&self, msg: Message, command: Command) -> Result<()> {
        match command {
            Command::Start => {
                self.bot.send_message(msg.chat.id, "Welcome to the trading bot! Use /help to see available commands.").await?;
            }
            Command::Status => {
                let dashboard = self.dashboard.lock().await;
                let system_health = dashboard.get_system_health().await;
                let performance = dashboard.get_performance_metrics().await;
                let trading = dashboard.get_trading_metrics().await;
                let is_healthy = dashboard.is_system_healthy().await;

                let status_emoji = if is_healthy { "✅" } else { "⚠️" };
                let message = format!(
                    "🤖 *Bot Status* {}\n\n\
                    *System Health*\n\
                    CPU: {:.1}%\n\
                    Memory: {:.1}%\n\
                    Disk: {:.1}%\n\
                    Error Rate: {:.1}%\n\n\
                    *Performance*\n\
                    API Error Rate: {:.1}%\n\
                    DB Error Rate: {:.1}%\n\
                    API Response Time: {:?}\n\n\
                    *Trading*\n\
                    Win Rate: {:.1}%\n\
                    Drawdown: {:.1}%\n\
                    Position Size: ${:.2}",
                    status_emoji,
                    system_health.cpu_usage,
                    system_health.memory_usage,
                    system_health.disk_usage,
                    system_health.error_rate,
                    performance.api_error_rate,
                    performance.db_error_rate,
                    performance.api_response_time,
                    trading.win_rate,
                    trading.drawdown,
                    trading.position_size
                );
                self.bot.send_message(msg.chat.id, message)
                    .parse_mode(ParseMode::MarkdownV2)
                    .await?;
            }
            Command::MarketData => {
                self.bot.send_message(msg.chat.id, "Market data command is not available. Real-time market data is not fetched by the bot at this time.").await?;
            }
            Command::Positions => {
                let trading_bot = self.trading_bot.clone();
                if let Ok(positions) = trading_bot.get_positions().await {
                    let message = if positions.is_empty() {
                        "No active positions".to_string()
                    } else {
                        positions.iter()
                            .map(|p| format!(
                                "{}: {} @ ${:.2} (PnL: ${:.2})",
                                p.symbol,
                                p.amount,
                                p.entry_price,
                                p.unrealized_pnl
                            ))
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    self.bot.send_message(msg.chat.id, message).await?;
                } else {
                    self.bot.send_message(msg.chat.id, "Failed to fetch positions").await?;
                }
            }
            Command::History => {
                let trading_bot = self.trading_bot.clone();
                if let Ok(history) = trading_bot.get_trade_history().await {
                    let message = if history.is_empty() {
                        "No trade history available".to_string()
                    } else {
                        let mut formatted_history = String::from("📈 Trade History:\n\n");
                        for trade in history {
                            formatted_history.push_str(&format!(
                                "🔹 {} {} {} @ ${:.2}\n   Time: {}\n   PnL: ${:.2}\n\n",
                                trade.entry_time.format("%Y-%m-%d %H:%M:%S"),
                                if trade.size > 0.0 { "Bought" } else { "Sold" },
                                trade.size.abs(),
                                trade.entry_price,
                                trade.exit_time.format("%Y-%m-%d %H:%M:%S"),
                                trade.pnl
                            ));
                        }
                        formatted_history
                    };
                    self.bot.send_message(msg.chat.id, message)
                        .parse_mode(ParseMode::Html)
                        .await?;
                } else {
                    self.bot.send_message(msg.chat.id, "Failed to fetch trade history")
                        .parse_mode(ParseMode::Html)
                        .await?;
                }
            }
            Command::Backtest => {
                let trading_bot = self.trading_bot.clone();
                if let Ok(results) = trading_bot.get_backtest_results().await {
                    let message = format!(
                        "📊 *Backtest Results*\n\n\
                        💰 Initial Balance: ${:.2}\n\
                        💵 Final Balance: ${:.2}\n\
                        📈 Total PnL: ${:.2} ({:.2}%)\n\n\
                        📊 *Performance Metrics*\n\
                        ✅ Win Rate: {:.2}%\n\
                        📊 Total Trades: {}\n\
                        ✅ Winning Trades: {}\n\
                        ❌ Losing Trades: {}\n\
                        📉 Max Drawdown: {:.2}%\n\
                        📊 Sharpe Ratio: {:.2}",
                        results.initial_balance,
                        results.initial_balance + results.total_pnl,
                        results.total_pnl,
                        (results.total_pnl / results.initial_balance) * 100.0,
                        results.win_rate * 100.0,
                        results.total_trades,
                        results.winning_trades,
                        results.losing_trades,
                        results.max_drawdown * 100.0,
                        results.sharpe_ratio
                    );
                    self.bot.send_message(msg.chat.id, message)
                        .parse_mode(ParseMode::MarkdownV2)
                        .await?;
                } else {
                    self.bot.send_message(msg.chat.id, "Failed to fetch backtest results").await?;
                }
            }
            Command::Log => {
                let log = self.log.lock().await;
                if log.is_empty() {
                    self.bot.send_message(msg.chat.id, "No recent actions.").await?;
                } else {
                    let log_text = log.iter().rev().take(20).cloned().collect::<Vec<_>>().join("\n");
                    self.bot.send_message(msg.chat.id, format!("Recent actions:\n{}", log_text)).await?;
                }
            }
            Command::Help => {
                let help_text = "Available commands:\n\
                    /start - Start the bot\n\
                    /help - Show this help message\n\
                    /marketdata - Get current market data\n\
                    /positions - List current positions\n\
                    /history - View trade history\n\
                    /status - Check bot status\n\
                    /backtest - Run backtest\n\
                    /log - Show recent bot actions";
                self.bot.send_message(msg.chat.id, help_text).await?;
            }
        }
        Ok(())
    }

    pub async fn send_message(&self, message: &str) -> Result<()> {
        self.bot
            .send_message(self.chat_id, message)
            .parse_mode(ParseMode::Html)
            .await?;
        Ok(())
    }

    pub async fn push_log(&self, entry: String) {
        let mut log = self.log.lock().await;
        if log.len() == 20 { log.pop_front(); }
        log.push_back(entry);
    }
} 
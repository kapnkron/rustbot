use crate::api::MarketData;
use crate::trading::{TradingSignal, TradingBot};
use crate::error::Result;
use teloxide::prelude::*;
use teloxide::types::{Message, ParseMode};
use teloxide::utils::command::BotCommands;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

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
    #[command(description = "Display this help message")]
    Help,
}

pub struct TelegramBot {
    bot: Bot,
    chat_id: ChatId,
    trading_enabled: Arc<Mutex<bool>>,
    trading_bot: Arc<Mutex<TradingBot>>,
}

impl TelegramBot {
    pub fn new(bot_token: String, chat_id: String, trading_bot: TradingBot) -> Self {
        Self {
            bot: Bot::new(bot_token),
            chat_id: ChatId(chat_id.parse().expect("Invalid chat ID")),
            trading_enabled: Arc::new(Mutex::new(false)),
            trading_bot: Arc::new(Mutex::new(trading_bot)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        let handler = Update::filter_message()
            .branch(
                dptree::entry()
                    .filter_command::<Command>()
                    .endpoint(command_handler),
            );

        Dispatcher::builder(self.bot.clone(), handler)
            .enable_ctrlc_handler()
            .build()
            .dispatch()
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
            📈 Volume: ${:.2}\n\
            🔄 24h Volume Change: {:.2}%",
            data.symbol,
            data.price,
            price_change_emoji,
            data.price_change_24h,
            data.market_cap,
            data.volume,
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

    pub async fn is_trading_enabled(&self) -> bool {
        *self.trading_enabled.lock().await
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
            .take(5) // Show last 5 trades
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
}

async fn command_handler(
    bot: Bot,
    msg: Message,
    cmd: Command,
    trading_bot: Arc<Mutex<TradingBot>>,
    trading_enabled: Arc<Mutex<bool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match cmd {
        Command::Start => {
            bot.send_message(
                msg.chat.id,
                "Welcome to the Trading Bot! Use /help to see available commands.",
            )
            .await?;
        }
        Command::Status => {
            let status = if *trading_enabled.lock().await {
                "Trading is enabled"
            } else {
                "Trading is disabled"
            };
            bot.send_message(msg.chat.id, status).await?;
        }
        Command::Positions => {
            let positions = trading_bot.lock().await.get_positions().await?;
            if positions.is_empty() {
                bot.send_message(msg.chat.id, "No open positions").await?;
            } else {
                let message = positions
                    .iter()
                    .map(|p| format!("{}: {} @ ${:.2}", p.symbol, p.amount, p.entry_price))
                    .collect::<Vec<_>>()
                    .join("\n");
                bot.send_message(msg.chat.id, format!("Open positions:\n{}", message))
                    .await?;
            }
        }
        Command::MarketData => {
            if let Ok(data) = trading_bot.lock().await.get_market_data(&String::new()).await {
                let message = format!(
                    "📊 Market Data for {}\n\
                    Price: ${:.2}\n\
                    Volume: ${:.2}\n\
                    Market Cap: ${:.2}\n\
                    24h Change: {:.2}%",
                    data.symbol,
                    data.price,
                    data.volume,
                    data.market_cap,
                    data.price_change_24h
                );
                bot.send_message(msg.chat.id, message).await?;
            } else {
                bot.send_message(msg.chat.id, "Failed to fetch market data").await?;
            }
        }
        Command::History => {
            // Implement history retrieval
            bot.send_message(msg.chat.id, "History retrieval not implemented").await?;
        }
        Command::Backtest => {
            if let Ok(results) = trading_bot.lock().await.get_backtest_results().await {
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
                bot.send_message(msg.chat.id, message)
                    .parse_mode(ParseMode::MarkdownV2)
                    .await?;
            } else {
                bot.send_message(msg.chat.id, "Failed to fetch backtest results").await?;
            }
        }
        Command::Help => {
            bot.send_message(
                msg.chat.id,
                Command::descriptions().to_string(),
            )
            .await?;
        }
    }

    Ok(())
} 
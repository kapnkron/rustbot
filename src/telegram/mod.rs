use crate::utils::error::Result;
use crate::trading::{MarketData, TradingSignal, TradingBot};
use teloxide::prelude::*;
use teloxide::types::{Message, ParseMode};
use teloxide::utils::command::BotCommands;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error};

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
        let message = format!(
            "📊 Market Data for {}\n\
            Price: ${:.2}\n\
            Volume: ${:.2}\n\
            Market Cap: ${:.2}\n\
            24h Change: {:.2}%\n\
            24h Volume Change: {:.2}%",
            data.symbol,
            data.price,
            data.volume,
            data.market_cap,
            data.price_change_24h,
            data.volume_change_24h
        );

        self.bot
            .send_message(self.chat_id, message)
            .parse_mode(ParseMode::Html)
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
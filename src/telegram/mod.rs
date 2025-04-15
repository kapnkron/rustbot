use crate::utils::error::Result;
use crate::trading::{MarketData, TradingSignal};
use teloxide::prelude::*;
use teloxide::types::{Message, ParseMode};
use teloxide::utils::command::BotCommands;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error};

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
pub enum Command {
    #[command(description = "Start the bot")]
    Start,
    #[command(description = "Get current status")]
    Status,
    #[command(description = "Get current positions")]
    Positions,
    #[command(description = "Get market data for a symbol")]
    Market(String),
    #[command(description = "Enable/disable trading")]
    Trading(bool),
    #[command(description = "Set risk level")]
    Risk(f64),
    #[command(description = "Get help")]
    Help,
}

pub struct TelegramBot {
    bot: Bot,
    chat_id: ChatId,
    trading_enabled: Arc<Mutex<bool>>,
}

impl TelegramBot {
    pub fn new(bot_token: String, chat_id: String) -> Self {
        Self {
            bot: Bot::new(bot_token),
            chat_id: ChatId(chat_id.parse().expect("Invalid chat ID")),
            trading_enabled: Arc::new(Mutex::new(false)),
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
            // TODO: Implement status command
            bot.send_message(msg.chat.id, "Status: Running").await?;
        }
        Command::Positions => {
            // TODO: Implement positions command
            bot.send_message(msg.chat.id, "No open positions").await?;
        }
        Command::Market(symbol) => {
            // TODO: Implement market data command
            bot.send_message(msg.chat.id, format!("Fetching data for {}", symbol))
                .await?;
        }
        Command::Trading(enabled) => {
            // TODO: Implement trading toggle
            bot.send_message(
                msg.chat.id,
                format!("Trading {}", if enabled { "enabled" } else { "disabled" }),
            )
            .await?;
        }
        Command::Risk(level) => {
            // TODO: Implement risk level setting
            bot.send_message(msg.chat.id, format!("Risk level set to {}", level))
                .await?;
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
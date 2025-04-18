use anyhow::Result;
use clap::Parser;
use log::{info, error};
use std::path::PathBuf;
use tokio::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

mod api;
mod config;
mod models;
mod services;
mod utils;
mod ml;
mod monitoring;
mod telegram;
mod security;
mod trading;
mod wallet;

use crate::config::Config;
use crate::trading::{TradingBot, MarketData, TradingSignal};
use crate::api::MarketDataCollector;
use crate::telegram::TelegramBot;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    env_logger::Builder::new()
        .filter_level(if args.debug {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    // Load configuration
    let config = Config::load(&args.config)?;
    info!("Configuration loaded successfully");

    // Create channels for market data and trading signals
    let (market_data_tx, mut market_data_rx) = mpsc::channel(100);
    let (signal_tx, mut signal_rx) = mpsc::channel(100);

    // Initialize market data collector
    let market_data_collector = MarketDataCollector::new(
        config.api.coingecko_api_key.clone(),
        config.api.coinmarketcap_api_key.clone(),
        config.api.cryptodatadownload_api_key.clone(),
    );

    // Initialize trading bot
    let trading_bot = TradingBot::new(market_data_collector);

    // Initialize Telegram bot
    let telegram_bot = TelegramBot::new(
        config.telegram.bot_token.clone(),
        config.telegram.chat_id.clone(),
        trading_bot.clone(),
    );

    // Start Telegram bot
    let telegram_handle = tokio::spawn(async move {
        if let Err(e) = telegram_bot.start().await {
            error!("Telegram bot error: {}", e);
        }
    });

    // Start market data collection
    let market_data_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            for symbol in &config.trading.trading_pairs {
                match market_data_collector.collect_market_data(symbol).await {
                    Ok(data) => {
                        if let Err(e) = market_data_tx.send(data).await {
                            error!("Failed to send market data: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect market data for {}: {}", symbol, e);
                    }
                }
            }
        }
    });

    // Start signal processing
    let signal_handle = tokio::spawn(async move {
        while let Some(signal) = signal_rx.recv().await {
            if let Err(e) = trading_bot.execute_signal(signal).await {
                error!("Error executing signal: {}", e);
            }
        }
    });

    // Start market data processing
    let market_data_processor = tokio::spawn(async move {
        while let Some(data) = market_data_rx.recv().await {
            match trading_bot.process_market_data(data.into()).await {
                Ok(Some(signal)) => {
                    if let Err(e) = signal_tx.send(signal).await {
                        error!("Failed to send trading signal: {}", e);
                    }
                }
                Ok(None) => {
                    info!("No trading signal generated");
                }
                Err(e) => {
                    error!("Error processing market data: {}", e);
                }
            }
        }
    });

    // Wait for all tasks to complete
    tokio::try_join!(
        telegram_handle,
        market_data_handle,
        signal_handle,
        market_data_processor
    )?;

    Ok(())
}

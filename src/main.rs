use anyhow::Result;
use clap::Parser;
use log::{info, error};
use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tokio::time::interval;

use trading_bot::monitoring::dashboard::Dashboard;
use trading_bot::monitoring::thresholds::{ThresholdConfig, SystemThresholds, PerformanceThresholds, TradeThresholds, NotificationSettings};
use trading_bot::config::Config;
use trading_bot::trading::TradingBot;
use trading_bot::api::MarketDataCollector;
use trading_bot::ml::{TradingModel, ModelConfig, Predictor};
use trading_bot::telegram::TelegramBot;
use trading_bot::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    info!("Starting trading bot...");

    // Load configuration
    let config_path = cli.config.unwrap_or_else(|| "config/config.toml".into());
    let config = match Config::load(&config_path) {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            eprintln!("Failed to load configuration from {:?}: {}", config_path, e);
            return Err(anyhow::anyhow!("Configuration loading failed: {}", e).into());
        }
    };
    info!("Configuration loaded successfully.");

    // Initialize components
    let market_data_collector = Arc::new(MarketDataCollector::new(
        config.api.coingecko_api_key.clone(),
        config.api.coinmarketcap_api_key.clone(),
        config.api.cryptodatadownload_api_key.clone(),
    ));
    info!("Market data collector initialized.");

    // Create the actual TradingModel instance
    let model_config = ModelConfig::new(
        config.ml.architecture.clone(),
        config.ml.loss_function.clone(),
        config.ml.learning_rate,
        config.ml.model_path.clone(),
        config.ml.window_size,
        config.ml.min_data_points,
    )?;
    let trading_model = TradingModel::new(model_config)?;
    // Potentially load weights here if needed: trading_model.load(...) 
    let model_predictor: Arc<Mutex<dyn Predictor + Send>> = Arc::new(Mutex::new(trading_model));
    info!("Trading model initialized.");

    // Pass the model instance to TradingBot::new
    let trading_bot = Arc::new(TradingBot::new(market_data_collector.clone(), model_predictor, config.clone())?);
    info!("Trading bot initialized.");

    let telegram_bot = if config.telegram.enable_notifications {
        // Manually create a default ThresholdConfig
        let default_thresholds = ThresholdConfig {
            system_thresholds: SystemThresholds {
                cpu_usage: 90.0, memory_usage: 90.0, disk_usage: 95.0, error_rate: 10.0,
                api_timeout: Duration::from_secs(15), db_timeout: Duration::from_secs(5),
            },
            performance_thresholds: PerformanceThresholds {
                api_error_rate: 5.0, db_error_rate: 5.0,
                api_response_time: Duration::from_secs(2), db_query_time: Duration::from_secs(1),
                ml_inference_time: Duration::from_millis(500),
            },
            trade_thresholds: TradeThresholds {
                win_rate: 0.4, max_drawdown: 0.25, max_position_size: 10000.0, // Example values
                min_profit_per_trade: 0.0, max_loss_per_trade: 500.0, daily_loss_limit: 1000.0,
            },
            notification_settings: NotificationSettings {
                alert_cooldown: Duration::from_secs(300), max_alerts_per_hour: 10,
                alert_priority: HashMap::new(), // Empty priorities for default
            },
        };
        let dashboard = Dashboard::new(default_thresholds, Duration::from_secs(60));
        
        Some(TelegramBot::new(
            config.telegram.bot_token.clone(), 
            config.telegram.chat_id.clone(), 
            trading_bot.clone(),
            dashboard 
        ))
    } else {
        None
    };
    if telegram_bot.is_some() {
        info!("Telegram bot initialized.");
    }

    // Initialize Monitor (assuming Monitor::new signature is fixed)
    // let registry = Arc::new(prometheus::Registry::new());
    // let monitor = match Monitor::new(registry.clone()) { // Use updated Monitor::new
    //     Ok(m) => Arc::new(m),
    //     Err(e) => {
    //         eprintln!("Failed to initialize monitor: {}", e);
    //         exit(1);
    //     }
    // };
    // info!("Monitor initialized.");

    // Start the main trading loop
    let mut interval = interval(Duration::from_secs(60)); // Example: Check every 60 seconds
    info!("Starting main trading loop...");

    // Enable trading initially (or based on config/command line)
    trading_bot.enable_trading(true).await;

    loop {
        interval.tick().await;
        info!("Tick: Processing market data...");

        // Iterate over the configured trading pairs
        for symbol in &config.trading.trading_pairs { // Loop through config.trading.trading_pairs
            info!("Processing pair: {}", symbol);
            // Fetch market data for configured pairs
            match trading_bot.get_market_data(symbol).await {
                Ok(market_data) => {
                    info!("Fetched market data for {}: Price = {}", symbol, market_data.price);
                    
                    // Process data and generate signal
                    match trading_bot.process_market_data(market_data).await {
                        Ok(Some(signal)) => {
                            info!("Signal generated: {:?}", signal);
                            // Execute the signal using execute_swap
                            let slippage_bps = config.dex_trading.slippage_bps; // Read from config
                            if let Err(e) = trading_bot.execute_swap(signal, slippage_bps).await {
                                error!("Failed to execute swap: {}", e);
                                if let Some(tg) = &telegram_bot {
                                    let _ = tg.send_message(&format!("Failed to execute swap: {}", e)).await;
                                }
                            }
                        }
                        Ok(None) => {
                            info!("No actionable signal generated (Hold or low confidence).");
                        }
                        // Handle MLWarmupRequired error specifically
                        Err(trading_bot::error::Error::MLWarmupRequired(reason)) => {
                            info!("ML Model warmup required: {}", reason);
                            // Bot will keep trying on next tick
                        }
                        Err(e) => {
                            error!("Error processing market data for {}: {}", symbol, e);
                            if let Some(tg) = &telegram_bot {
                                let _ = tg.send_message(&format!("Error processing market data for {}: {}", symbol, e)).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to fetch market data for {}: {}", symbol, e);
                     if let Some(tg) = &telegram_bot {
                         let _ = tg.send_message(&format!("Failed to fetch market data for {}: {}", symbol, e)).await;
                     }
                }
            }
        } // End of loop over symbols
    }

    // Note: The loop above is infinite. Code below might not be reached without explicit exit/shutdown logic.
    // Ok(())
}

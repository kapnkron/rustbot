use anyhow::Result;
use clap::Parser;
use log::{info, error};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use chrono::{Utc, Datelike};

use trading_bot::config::Config;
// use trading_bot::api::MarketDataCollector; // Commented out, role might change
// use trading_bot::ml::{PythonPredictor, Predictor}; // PythonPredictor might be replaced
use trading_bot::cli::Cli;
use trading_bot::ml_api_adapter; // Added use statement
use trading_bot::monitoring::dashboard::Dashboard;
use trading_bot::monitoring::thresholds::ThresholdConfig;
use trading_bot::telegram::TelegramBot;
use trading_bot::trading::portfolio::{PortfolioManager, PortfolioConfig, PositionSide};

// Add a dummy model that implements Predictor
struct DummyPredictor;

#[async_trait::async_trait]
impl trading_bot::ml::Predictor for DummyPredictor {
    async fn predict(&mut self, _data: &trading_bot::trading::TradingMarketData) -> trading_bot::error::Result<trading_bot::ml::PredictionOutput> {
        Ok(trading_bot::ml::PredictionOutput::new(vec![0.5, 0.5], 0.5))
    }
}

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
            return Err(anyhow::anyhow!("Configuration loading failed: {}", e));
        }
    };
    info!("Configuration loaded successfully.");

    // Initialize ML API Client
    let ml_api_client = Arc::new(
        ml_api_adapter::MlApiClient::new(config.api.ml_api_base_url.clone())
    );
    info!("ML API Client initialized with base URL: {}", config.api.ml_api_base_url);

    // --- Remove temporary test for MlApiClient ---
    // --- End of temporary test ---

    // Initialize components
    /*
    let market_data_collector = Arc::new(MarketDataCollector::new(
        config.api.coingecko_api_key.clone(), // These fields no longer exist on config.api
        config.api.coinmarketcap_api_key.clone(),
        config.api.cryptodatadownload_api_key.clone(),
    ));
    info!("Market data collector initialized.");
    */

    // Create the PythonPredictor instance - This will be replaced by MlApiClient usage
    /*
    let python_predictor = PythonPredictor::new(
        config.ml.model_path.clone(),        // Path to the .pt or .onnx model file
        config.ml.scaler_path.clone(),       // Path to the .pkl scaler file
        config.ml.python_script_path.clone() // Path to the Python inference script
    );
    let model_predictor: Arc<Mutex<dyn Predictor + Send>> = Arc::new(Mutex::new(python_predictor));
    info!("Python predictor initialized.");
    */

    // Wrap the collector in Arc<Mutex<M>>
    // let collector_mutex = Arc::new(Mutex::new(market_data_collector.as_ref().clone())); // market_data_collector commented out

    // Initialize TradingBot with dummy model
    let dummy_model: Arc<tokio::sync::Mutex<dyn trading_bot::ml::Predictor + Send>> = Arc::new(tokio::sync::Mutex::new(DummyPredictor));
    let trading_bot = Arc::new(trading_bot::trading::TradingBot::new(dummy_model, config.clone(), config.paper_trading)?);
    trading_bot.enable_trading(true).await;

    // --- TelegramBot Initialization ---
    // Minimal dummy dashboard config for now
    let default_threshold_config = ThresholdConfig {
        system_thresholds: trading_bot::monitoring::thresholds::SystemThresholds {
            cpu_usage: 90.0,
            memory_usage: 90.0,
            disk_usage: 90.0,
            error_rate: 10.0,
            api_timeout: std::time::Duration::from_secs(5),
            db_timeout: std::time::Duration::from_secs(5),
        },
        performance_thresholds: trading_bot::monitoring::thresholds::PerformanceThresholds {
            api_error_rate: 5.0,
            db_error_rate: 5.0,
            api_response_time: std::time::Duration::from_secs(1),
            db_query_time: std::time::Duration::from_secs(1),
            ml_inference_time: std::time::Duration::from_secs(1),
        },
        trade_thresholds: trading_bot::monitoring::thresholds::TradeThresholds {
            win_rate: 50.0,
            max_drawdown: 20.0,
            max_position_size: 1000.0,
            min_profit_per_trade: 10.0,
            max_loss_per_trade: 100.0,
            daily_loss_limit: 1000.0,
        },
        notification_settings: trading_bot::monitoring::thresholds::NotificationSettings {
            alert_cooldown: std::time::Duration::from_secs(300),
            max_alerts_per_hour: 12,
            alert_priority: std::collections::HashMap::new(),
        },
    };
    let dashboard_update_interval = std::time::Duration::from_secs(60);
    let dashboard = Dashboard::new(default_threshold_config, dashboard_update_interval);
    let telegram_bot = Arc::new(TelegramBot::new(
        config.telegram.bot_token.clone(),
        config.telegram.chat_id.clone(),
        trading_bot.clone(),
        dashboard,
    ));
    // TODO: Replace dummy dashboard with real metrics/dashboard in the future

    // --- Risk Management: PortfolioManager Setup ---
    let portfolio_config = PortfolioConfig {
        initial_balance: 2000.0, // TODO: Load from config or wallet
        base_currency: "USD".to_string(),
        max_positions: 3,
        position_size_limit: 0.2, // Max 20% of portfolio per position
        risk_per_trade: 0.01,     // 1% risk per trade
        stop_loss_percentage: 0.05, // 5% stop-loss
    };
    let portfolio = Arc::new(PortfolioManager::new(portfolio_config));

    // --- Risk Management: Daily PnL Tracking ---
    let mut daily_realized_pnl: f64 = 0.0;
    let mut last_pnl_day = Utc::now().day();
    let daily_loss_limit = 1000.0; // TODO: Load from config
    let take_profit_percentage = 0.10; // 10% take-profit (optional, can be made configurable)
    let mut trading_paused_for_day = false;

    // Start the main trading loop
    let mut interval = interval(Duration::from_secs(60)); // Example: Check every 60 seconds
    info!("Starting main trading loop...");

    loop {
        info!("Main loop: Top of loop, awaiting interval tick...");
        interval.tick().await;
        info!("Main loop: Interval ticked.");

        let now = Utc::now();
        // Reset daily PnL at the start of a new day
        if now.day() != last_pnl_day {
            info!("Main loop: Resetting daily PnL.");
            daily_realized_pnl = 0.0;
            last_pnl_day = now.day();
            trading_paused_for_day = false;
        }

        info!("Main loop: Checking stop-loss/take-profit...");
        let open_positions = portfolio.get_positions().await;
        for pos in open_positions.iter() {
            let stop_loss = pos.entry_price * (1.0 - portfolio.stop_loss_percentage());
            let take_profit = pos.entry_price * (1.0 + take_profit_percentage);
            let current_price = pos.current_price;
            let mut closed = false;
            let mut reason = String::new();
            if current_price <= stop_loss {
                // Stop-loss triggered
                match portfolio.close_position(&pos.symbol, current_price).await {
                    Ok(pnl) => {
                        daily_realized_pnl += pnl;
                        reason = format!("Stop-loss hit for {} at ${:.2}, PnL: ${:.2}", pos.symbol, current_price, pnl);
                        closed = true;
                    },
                    Err(e) => info!("Failed to close position at stop-loss: {}", e),
                }
            } else if current_price >= take_profit {
                // Take-profit triggered
                match portfolio.close_position(&pos.symbol, current_price).await {
                    Ok(pnl) => {
                        daily_realized_pnl += pnl;
                        reason = format!("Take-profit hit for {} at ${:.2}, PnL: ${:.2}", pos.symbol, current_price, pnl);
                        closed = true;
                    },
                    Err(e) => info!("Failed to close position at take-profit: {}", e),
                }
            }
            if closed {
                info!("{}", reason);
                // Send Telegram notification
                if let Err(e) = telegram_bot.send_alert(&reason).await {
                    error!("Failed to send Telegram alert: {}", e);
                }
                // Log the event
                telegram_bot.push_log(reason).await;
            }
        }
        info!("Main loop: Finished stop-loss/take-profit check.");

        info!("Main loop: Checking daily loss limit...");
        if daily_realized_pnl <= -daily_loss_limit && !trading_paused_for_day {
            trading_paused_for_day = true;
            let alert_msg = format!("Daily loss limit of ${:.2} exceeded (PnL: ${:.2}). Trading paused for the day.", daily_loss_limit, daily_realized_pnl);
            info!("{}", alert_msg);
            if let Err(e) = telegram_bot.send_alert(&alert_msg).await {
                error!("Failed to send Telegram alert: {}", e);
            }
            telegram_bot.push_log(alert_msg).await;
        }
        if trading_paused_for_day {
            info!("Main loop: Trading paused for the day, continuing to next tick.");
            continue;
        }
        info!("Main loop: Finished daily loss limit check.");

        info!("Tick: Requesting predictions from ML API...");

        // Request predictions for dynamic tokens (let Python API decide)
        let predict_request = ml_api_adapter::PredictRequest {
            tokens: None, // Let Python API select tokens (e.g., top gainers/losers)
            force_feature_recalculation: false,
        };
        match ml_api_client.get_predictions(&predict_request).await {
            Ok(response) => {
                info!("Received predictions: {:#?}", response);
                for signal in response.signals.iter() {
                    let token_address = signal.token_address.as_deref().unwrap_or("<unknown>");
                    let current_price = signal.current_price_actual.unwrap_or(f64::NAN); // Use actual current price
                    let confidence = signal.signal_strength.unwrap_or(0.0); // Directly use confidence
                    
                    // Determine action based on the 'action' string from the API signal
                    let action_str = signal.action.as_deref().unwrap_or("Hold");
                    let bot_signal_type = match action_str {
                        "Buy" => trading_bot::trading::SignalType::Buy,
                        "Sell" => trading_bot::trading::SignalType::Sell,
                        _ => trading_bot::trading::SignalType::Hold, // Default to Hold for "Hold" or "Unknown"
                    };

                    // Log the processed signal details from API
                    info!(
                        "Processed API Signal: token_address={}, action='{}', current_price={:.6}, confidence={:.4}, raw_probs={:?}",
                        token_address, 
                        action_str,
                        current_price, 
                        confidence,
                        signal.raw_prediction.as_deref().unwrap_or_default()
                    );

                    // --- Risk Management Logic (remains largely the same, uses current_price) ---
                    let open_positions_count = portfolio.get_positions().await.len(); // Fetch once
                    if bot_signal_type != trading_bot::trading::SignalType::Hold && open_positions_count >= portfolio.max_positions() {
                        info!(
                            "Max open positions ({}) reached. Skipping {} for {}.", 
                            portfolio.max_positions(), 
                            action_str, 
                            token_address
                        );
                        continue;
                    }

                    // --- Position Sizing (uses current_price) ---
                    let position_size_usd = if bot_signal_type != trading_bot::trading::SignalType::Hold {
                        let stop_loss_price = current_price * (1.0 - portfolio.stop_loss_percentage());
                        match portfolio.calculate_position_size(&token_address.to_string(), current_price, stop_loss_price).await {
                            Ok(size) if size > 0.0 => size,
                            _ => {
                                info!("Calculated position size is zero or error for {}. Skipping trade.", token_address);
                                continue; // Skip if position size can't be determined for Buy/Sell
                            }
                        }
                    } else {
                        0.0 // No position size needed for Hold
                    };

                    // --- Track positions before trade attempt (for Buy/Sell) ---
                    let positions_before = if bot_signal_type != trading_bot::trading::SignalType::Hold {
                        portfolio.get_positions().await
                    } else {
                        Vec::new() // Not needed for Hold
                    };
                    let pos_before_trade = positions_before.iter().find(|p| p.symbol == token_address);

                    // --- Open/Close Position Logic (based on bot_signal_type) ---
                    let mut trade_executed_successfully = false;
                    let mut actual_trade_action_for_notification = bot_signal_type.clone(); // For precise notification

                    if bot_signal_type == trading_bot::trading::SignalType::Buy {
                        if position_size_usd > 0.0 {
                            let quantity = position_size_usd / current_price;
                            match portfolio.open_position(
                                token_address.to_string(),
                                quantity,
                                current_price,
                                PositionSide::Long,
                            ).await {
                                Ok(_) => {
                                    info!("Portfolio: Opened position for {} qty {} @ ${:.2}", token_address, quantity, current_price);
                                    trade_executed_successfully = true;
                                }
                                Err(e) => {
                                    error!("Portfolio: Failed to open position for {}: {}", token_address, e);
                                }
                            }
                        } else {
                            info!("Skipping Buy for {} due to zero position size.", token_address);
                        }
                    } else if bot_signal_type == trading_bot::trading::SignalType::Sell {
                        // Check if there is an existing position to close for this token
                        if portfolio.get_positions().await.iter().any(|p| p.symbol == token_address) {
                            match portfolio.close_position(token_address, current_price).await {
                                Ok(pnl) => {
                                    info!("Portfolio: Closed position for {} at ${:.2}, PnL: ${:.2}", token_address, current_price, pnl);
                                    daily_realized_pnl += pnl; // Update daily PnL
                                    trade_executed_successfully = true;
                                }
                                Err(e) => {
                                    error!("Portfolio: Failed to close position for {}: {}", token_address, e);
                                }
                            }
                        } else {
                            info!("Sell signal for {}, but no open position found in portfolio. Skipping close.", token_address);
                            actual_trade_action_for_notification = trading_bot::trading::SignalType::Hold; // No actual trade happened
                        }
                    }

                    // --- Real Trade Execution via TradingBot (only if portfolio action was successful for paper trading) ---
                    // And only if the signal is Buy or Sell
                    if trade_executed_successfully && (bot_signal_type == trading_bot::trading::SignalType::Buy || bot_signal_type == trading_bot::trading::SignalType::Sell) {
                        let trading_signal_for_bot = trading_bot::trading::TradingSignal {
                            symbol: token_address.to_string(),
                            signal_type: bot_signal_type.clone(), // Use the action determined from API
                            confidence, // Use confidence from API
                            price: current_price, // Use current price from API
                        };
                        let slippage_bps = config.dex_trading.slippage_bps;
                        match trading_bot.execute_swap(trading_signal_for_bot.clone(), slippage_bps).await {
                            Ok(_) => {
                                info!("TradingBot: Swap execution successful for {:?} {}", bot_signal_type, token_address);
                                // Notification logic moved below, based on actual portfolio change
                            }
                            Err(e) => {
                                error!("TradingBot: Swap execution failed for {:?} {}: {}", bot_signal_type, token_address, e);
                                // Even if swap fails, portfolio might have changed in paper trading, so notification logic below still applies
                                // Send a specific alert for swap failure if needed
                                let alert_msg = format!("Swap execution failed for {:?} {}: {}", bot_signal_type, token_address, e);
                                if let Err(tel_e) = telegram_bot.send_alert(&alert_msg).await {
                                    error!("Failed to send Telegram alert for swap failure: {}", tel_e);
                                }
                                telegram_bot.push_log(alert_msg).await;
                            }
                        }
                    } else if bot_signal_type == trading_bot::trading::SignalType::Hold {
                        info!("Hold signal for {}. No trade action.", token_address);
                    }

                    // --- Notification Logic (based on actual portfolio change) ---
                    if bot_signal_type != trading_bot::trading::SignalType::Hold { // Only consider for Buy/Sell signals
                        let positions_after = portfolio.get_positions().await;
                        let pos_after_trade = positions_after.iter().find(|p| p.symbol == token_address);

                        let position_opened = pos_before_trade.is_none() && pos_after_trade.is_some();
                        let position_closed = pos_before_trade.is_some() && pos_after_trade.is_none();
                        // TODO: Add logic for position size changed if desired

                        if position_opened || position_closed {
                            let notification_action_str = if position_opened { "OPENED" } else { "CLOSED" };
                            // Use actual_trade_action_for_notification for the signal type in notification
                            // to reflect what the portfolio did (e.g. might be Hold if a Sell was attempted on no position)
                            let final_signal_for_notification = trading_bot::trading::TradingSignal {
                                symbol: token_address.to_string(),
                                signal_type: actual_trade_action_for_notification.clone(), 
                                confidence, 
                                price: current_price,
                            };

                            if final_signal_for_notification.signal_type != trading_bot::trading::SignalType::Hold {
                                if let Err(e) = telegram_bot.send_trading_signal(&final_signal_for_notification).await {
                                    error!("Failed to send Telegram trade notification: {}", e);
                                }
                                let log_entry = format!(
                                    "{} {} at {:.6} (api_action: {}, conf: {:.2}%)", 
                                    notification_action_str, 
                                    token_address, 
                                    current_price, 
                                    action_str, // Log the original action from API for clarity
                                    confidence * 100.0
                                );
                                telegram_bot.push_log(log_entry).await;
                            } else {
                                info!("Portfolio state implies Hold for {}. No trade notification sent for this signal cycle.", token_address);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to get predictions from ML API: {}", e);
                // Optionally send Telegram alert for error
                let alert_msg = format!("Failed to get predictions from ML API: {}", e);
                if let Err(e) = telegram_bot.send_alert(&alert_msg).await {
                    error!("Failed to send Telegram alert: {}", e);
                }
                // Log the ML API error
                let log_entry = format!("ML API ERROR: {}", e);
                telegram_bot.push_log(log_entry).await;
            }
        }
        // TODO: Log any other important events as needed
    }

    // Note: The loop above is infinite. Code below might not be reached without explicit exit/shutdown logic.
    // Ok(())
}

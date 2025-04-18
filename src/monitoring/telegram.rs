use crate::error::Result;
use serde::{Deserialize, Serialize};
use teloxide::prelude::*;
use teloxide::types::ParseMode;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub bot_token: String,
    pub chat_id: String,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub error_rate: f64,
    pub api_error_rate: f64,
    pub db_error_rate: f64,
    pub trade_win_rate: f64,
    pub ml_accuracy: f64,
}

pub struct TelegramNotifier {
    config: TelegramConfig,
    bot: Bot,
    is_initialized: bool,
}

impl TelegramNotifier {
    pub fn new(config: TelegramConfig) -> Self {
        Self {
            bot: Bot::new(&config.bot_token),
            config,
            is_initialized: false,
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        // Test the bot connection
        let me = self.bot.get_me().await?;
        info!("Telegram bot initialized: @{}", me.username);
        self.is_initialized = true;
        Ok(())
    }

    pub async fn send_alert(&self, message: &str) -> Result<()> {
        if !self.is_initialized {
            return Err(anyhow::anyhow!("Telegram notifier not initialized"));
        }

        self.bot
            .send_message(&self.config.chat_id, message)
            .parse_mode(ParseMode::MarkdownV2)
            .await?;

        Ok(())
    }

    pub async fn send_health_alert(&self, metrics: &crate::monitoring::health::HealthMetrics) -> Result<()> {
        if metrics.cpu_usage > self.config.alert_thresholds.cpu_usage {
            self.send_alert(&format!(
                "⚠️ *High CPU Usage Alert*\nCPU Usage: {:.2}%",
                metrics.cpu_usage
            )).await?;
        }

        if metrics.memory_usage > self.config.alert_thresholds.memory_usage {
            self.send_alert(&format!(
                "⚠️ *High Memory Usage Alert*\nMemory Usage: {:.2}%",
                metrics.memory_usage
            )).await?;
        }

        if metrics.error_rate > self.config.alert_thresholds.error_rate {
            self.send_alert(&format!(
                "⚠️ *High Error Rate Alert*\nError Rate: {:.2}%",
                metrics.error_rate
            )).await?;
        }

        if !metrics.api_status {
            self.send_alert("⚠️ *API Connection Lost*").await?;
        }

        if !metrics.db_status {
            self.send_alert("⚠️ *Database Connection Lost*").await?;
        }

        if !metrics.trading_status {
            self.send_alert("⚠️ *Trading Bot Stopped*").await?;
        }

        Ok(())
    }

    pub async fn send_performance_alert(&self, metrics: &crate::monitoring::performance::PerformanceMetrics) -> Result<()> {
        if metrics.api_metrics.error_rate > self.config.alert_thresholds.api_error_rate {
            self.send_alert(&format!(
                "⚠️ *High API Error Rate*\nError Rate: {:.2}%",
                metrics.api_metrics.error_rate
            )).await?;
        }

        if metrics.db_metrics.error_rate > self.config.alert_thresholds.db_error_rate {
            self.send_alert(&format!(
                "⚠️ *High Database Error Rate*\nError Rate: {:.2}%",
                metrics.db_metrics.error_rate
            )).await?;
        }

        if metrics.trade_metrics.win_rate < self.config.alert_thresholds.trade_win_rate {
            self.send_alert(&format!(
                "⚠️ *Low Trading Win Rate*\nWin Rate: {:.2}%",
                metrics.trade_metrics.win_rate
            )).await?;
        }

        if metrics.ml_metrics.prediction_accuracy < self.config.alert_thresholds.ml_accuracy {
            self.send_alert(&format!(
                "⚠️ *Low ML Model Accuracy*\nAccuracy: {:.2}%",
                metrics.ml_metrics.prediction_accuracy * 100.0
            )).await?;
        }

        Ok(())
    }

    pub async fn send_trade_notification(&self, trade_id: &str, success: bool, profit: f64) -> Result<()> {
        let emoji = if success { "✅" } else { "❌" };
        self.send_alert(&format!(
            "{} *Trade Completed*\nID: {}\nProfit: {:.2}",
            emoji, trade_id, profit
        )).await?;
        Ok(())
    }

    pub async fn send_daily_summary(
        &self,
        health_metrics: &crate::monitoring::health::HealthMetrics,
        perf_metrics: &crate::monitoring::performance::PerformanceMetrics,
    ) -> Result<()> {
        let message = format!(
            "📊 *Daily Summary*\n\n\
            *System Health*\n\
            CPU Usage: {:.2}%\n\
            Memory Usage: {:.2}%\n\
            Error Rate: {:.2}%\n\n\
            *Trading Performance*\n\
            Total Trades: {}\n\
            Win Rate: {:.2}%\n\
            Total Profit: {:.2}\n\n\
            *API Performance*\n\
            Requests: {}\n\
            Error Rate: {:.2}%\n\
            Avg Response Time: {:.2?}\n\n\
            *ML Performance*\n\
            Predictions: {}\n\
            Accuracy: {:.2}%",
            health_metrics.cpu_usage,
            health_metrics.memory_usage,
            health_metrics.error_rate,
            perf_metrics.trade_metrics.total_trades,
            perf_metrics.trade_metrics.win_rate,
            perf_metrics.trade_metrics.total_profit,
            perf_metrics.api_metrics.total_requests,
            perf_metrics.api_metrics.error_rate,
            perf_metrics.api_metrics.average_response_time,
            perf_metrics.ml_metrics.total_predictions,
            perf_metrics.ml_metrics.prediction_accuracy * 100.0
        );

        self.send_alert(&message).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telegram_notifier() -> Result<()> {
        let config = TelegramConfig {
            bot_token: "test_token".to_string(),
            chat_id: "test_chat".to_string(),
            alert_thresholds: AlertThresholds {
                cpu_usage: 90.0,
                memory_usage: 90.0,
                error_rate: 10.0,
                api_error_rate: 5.0,
                db_error_rate: 5.0,
                trade_win_rate: 50.0,
                ml_accuracy: 80.0,
            },
        };

        let mut notifier = TelegramNotifier::new(config);
        assert!(!notifier.is_initialized);
        
        // Note: This test requires a real Telegram bot token to work
        // For testing purposes, we'll just check that the notifier initializes correctly
        assert_eq!(notifier.config.bot_token, "test_token");
        assert_eq!(notifier.config.chat_id, "test_chat");
        
        Ok(())
    }
} 
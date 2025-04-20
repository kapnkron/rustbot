use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub system_thresholds: SystemThresholds,
    pub performance_thresholds: PerformanceThresholds,
    pub trade_thresholds: TradeThresholds,
    pub notification_settings: NotificationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemThresholds {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub error_rate: f64,
    pub api_timeout: Duration,
    pub db_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub api_error_rate: f64,
    pub db_error_rate: f64,
    pub api_response_time: Duration,
    pub db_query_time: Duration,
    pub ml_inference_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeThresholds {
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub max_position_size: f64,
    pub min_profit_per_trade: f64,
    pub max_loss_per_trade: f64,
    pub daily_loss_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub alert_cooldown: Duration,
    pub max_alerts_per_hour: u32,
    pub alert_priority: HashMap<String, AlertPriority>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AlertPriority {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ThresholdManager {
    config: ThresholdConfig,
    last_alerts: HashMap<String, (std::time::Instant, u32)>,
}

impl ThresholdManager {
    pub fn new(config: ThresholdConfig) -> Self {
        Self {
            config,
            last_alerts: HashMap::new(),
        }
    }

    pub fn validate_config(&self) -> Result<()> {
        // Validate system thresholds
        if self.config.system_thresholds.cpu_usage > 100.0 {
            return Err(anyhow::anyhow!("CPU usage threshold cannot exceed 100%").into());
        }
        if self.config.system_thresholds.memory_usage > 100.0 {
            return Err(anyhow::anyhow!("Memory usage threshold cannot exceed 100%").into());
        }
        if self.config.system_thresholds.disk_usage > 100.0 {
            return Err(anyhow::anyhow!("Disk usage threshold cannot exceed 100%").into());
        }

        // Validate performance thresholds
        if self.config.performance_thresholds.api_error_rate > 100.0 {
            return Err(anyhow::anyhow!("API error rate threshold cannot exceed 100%").into());
        }
        if self.config.performance_thresholds.db_error_rate > 100.0 {
            return Err(anyhow::anyhow!("Database error rate threshold cannot exceed 100%").into());
        }

        // Validate trade thresholds
        if self.config.trade_thresholds.win_rate > 100.0 {
            return Err(anyhow::anyhow!("Win rate threshold cannot exceed 100%").into());
        }
        if self.config.trade_thresholds.max_drawdown > 100.0 {
            return Err(anyhow::anyhow!("Max drawdown threshold cannot exceed 100%").into());
        }

        Ok(())
    }

    pub fn check_system_thresholds(&self, metrics: &crate::monitoring::health::HealthMetrics) -> Vec<String> {
        let mut alerts = Vec::new();

        if metrics.cpu_usage > self.config.system_thresholds.cpu_usage {
            alerts.push(format!("High CPU usage: {:.2}%", metrics.cpu_usage));
        }

        if metrics.memory_usage > self.config.system_thresholds.memory_usage {
            alerts.push(format!("High memory usage: {:.2}%", metrics.memory_usage));
        }

        if metrics.error_rate > self.config.system_thresholds.error_rate {
            alerts.push(format!("High error rate: {:.2}%", metrics.error_rate));
        }

        if !metrics.api_status {
            alerts.push("API connection lost".to_string());
        }

        if !metrics.db_status {
            alerts.push("Database connection lost".to_string());
        }

        alerts
    }

    pub fn check_performance_thresholds(&self, metrics: &crate::monitoring::performance::PerformanceMetrics) -> Vec<String> {
        let mut alerts = Vec::new();

        if metrics.api_metrics.error_rate > self.config.performance_thresholds.api_error_rate {
            alerts.push(format!("High API error rate: {:.2}%", metrics.api_metrics.error_rate));
        }

        if metrics.db_metrics.error_rate > self.config.performance_thresholds.db_error_rate {
            alerts.push(format!("High database error rate: {:.2}%", metrics.db_metrics.error_rate));
        }

        if metrics.api_metrics.average_response_time > self.config.performance_thresholds.api_response_time {
            alerts.push(format!("Slow API response time: {:?}", metrics.api_metrics.average_response_time));
        }

        if metrics.db_metrics.average_query_time > self.config.performance_thresholds.db_query_time {
            alerts.push(format!("Slow database query time: {:?}", metrics.db_metrics.average_query_time));
        }

        if metrics.ml_metrics.average_inference_time > self.config.performance_thresholds.ml_inference_time {
            alerts.push(format!("Slow ML inference time: {:?}", metrics.ml_metrics.average_inference_time));
        }

        alerts
    }

    pub fn check_trade_thresholds(&self, metrics: &crate::monitoring::performance::PerformanceMetrics) -> Vec<String> {
        let mut alerts = Vec::new();

        if metrics.trade_metrics.win_rate < self.config.trade_thresholds.win_rate {
            alerts.push(format!("Low win rate: {:.2}%", metrics.trade_metrics.win_rate));
        }

        let drawdown = metrics.trade_metrics.total_loss / (metrics.trade_metrics.total_profit + metrics.trade_metrics.total_loss) * 100.0;
        if drawdown > self.config.trade_thresholds.max_drawdown {
            alerts.push(format!("High drawdown: {:.2}%", drawdown));
        }

        if metrics.trade_metrics.total_loss > self.config.trade_thresholds.daily_loss_limit {
            alerts.push(format!("Daily loss limit exceeded: {:.2}", metrics.trade_metrics.total_loss));
        }

        alerts
    }

    pub fn should_send_alert(&mut self, alert_type: &str) -> bool {
        let now = std::time::Instant::now();
        let (last_alert_time, alert_count) = self.last_alerts.entry(alert_type.to_string())
            .or_insert((now, 0));

        // Check if we're within the cooldown period
        if now.duration_since(*last_alert_time) < self.config.notification_settings.alert_cooldown {
            return false;
        }

        // Check if we've exceeded the maximum alerts per hour
        if *alert_count >= self.config.notification_settings.max_alerts_per_hour {
            return false;
        }

        // Update the alert count and time
        *last_alert_time = now;
        *alert_count += 1;

        true
    }

    pub fn get_alert_priority(&self, alert_type: &str) -> AlertPriority {
        self.config.notification_settings.alert_priority
            .get(alert_type)
            .cloned()
            .unwrap_or(AlertPriority::Medium)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_threshold_manager() -> Result<()> {
        let config = ThresholdConfig {
            system_thresholds: SystemThresholds {
                cpu_usage: 90.0,
                memory_usage: 90.0,
                disk_usage: 90.0,
                error_rate: 10.0,
                api_timeout: Duration::from_secs(5),
                db_timeout: Duration::from_secs(5),
            },
            performance_thresholds: PerformanceThresholds {
                api_error_rate: 5.0,
                db_error_rate: 5.0,
                api_response_time: Duration::from_secs(1),
                db_query_time: Duration::from_secs(1),
                ml_inference_time: Duration::from_secs(1),
            },
            trade_thresholds: TradeThresholds {
                win_rate: 50.0,
                max_drawdown: 20.0,
                max_position_size: 1000.0,
                min_profit_per_trade: 10.0,
                max_loss_per_trade: 100.0,
                daily_loss_limit: 1000.0,
            },
            notification_settings: NotificationSettings {
                alert_cooldown: Duration::from_secs(300),
                max_alerts_per_hour: 12,
                alert_priority: HashMap::new(),
            },
        };

        let manager = ThresholdManager::new(config);
        assert!(manager.validate_config().is_ok());
        
        Ok(())
    }
} 
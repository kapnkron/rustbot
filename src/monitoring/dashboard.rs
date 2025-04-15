use crate::monitoring::thresholds::{ThresholdManager, ThresholdConfig};
use crate::monitoring::health::HealthMetrics;
use crate::monitoring::performance::PerformanceMetrics;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub system_health: SystemHealth,
    pub performance: Performance,
    pub trading: Trading,
    pub alerts: Vec<Alert>,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub error_rate: f64,
    pub api_status: bool,
    pub db_status: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Performance {
    pub api_error_rate: f64,
    pub db_error_rate: f64,
    pub api_response_time: Duration,
    pub db_query_time: Duration,
    pub ml_inference_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trading {
    pub win_rate: f64,
    pub drawdown: f64,
    pub total_profit: f64,
    pub total_loss: f64,
    pub position_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub message: String,
    pub priority: String,
    pub timestamp: Instant,
    pub resolved: bool,
}

pub struct Dashboard {
    metrics: Arc<RwLock<DashboardMetrics>>,
    threshold_manager: Arc<ThresholdManager>,
    update_interval: Duration,
}

impl Dashboard {
    pub fn new(config: ThresholdConfig, update_interval: Duration) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(DashboardMetrics {
                system_health: SystemHealth {
                    cpu_usage: 0.0,
                    memory_usage: 0.0,
                    disk_usage: 0.0,
                    error_rate: 0.0,
                    api_status: true,
                    db_status: true,
                },
                performance: Performance {
                    api_error_rate: 0.0,
                    db_error_rate: 0.0,
                    api_response_time: Duration::from_secs(0),
                    db_query_time: Duration::from_secs(0),
                    ml_inference_time: Duration::from_secs(0),
                },
                trading: Trading {
                    win_rate: 0.0,
                    drawdown: 0.0,
                    total_profit: 0.0,
                    total_loss: 0.0,
                    position_size: 0.0,
                },
                alerts: Vec::new(),
                last_updated: Instant::now(),
            })),
            threshold_manager: Arc::new(ThresholdManager::new(config)),
            update_interval,
        }
    }

    pub async fn update_metrics(&self, health_metrics: &HealthMetrics, perf_metrics: &PerformanceMetrics) {
        let mut metrics = self.metrics.write().await;
        
        // Update system health
        metrics.system_health.cpu_usage = health_metrics.cpu_usage;
        metrics.system_health.memory_usage = health_metrics.memory_usage;
        metrics.system_health.disk_usage = health_metrics.disk_usage;
        metrics.system_health.error_rate = health_metrics.error_rate;
        metrics.system_health.api_status = health_metrics.api_status;
        metrics.system_health.db_status = health_metrics.db_status;

        // Update performance metrics
        metrics.performance.api_error_rate = perf_metrics.api_metrics.error_rate;
        metrics.performance.db_error_rate = perf_metrics.db_metrics.error_rate;
        metrics.performance.api_response_time = perf_metrics.api_metrics.average_response_time;
        metrics.performance.db_query_time = perf_metrics.db_metrics.average_query_time;
        metrics.performance.ml_inference_time = perf_metrics.ml_metrics.average_inference_time;

        // Update trading metrics
        metrics.trading.win_rate = perf_metrics.trade_metrics.win_rate;
        metrics.trading.drawdown = perf_metrics.trade_metrics.total_loss / 
            (perf_metrics.trade_metrics.total_profit + perf_metrics.trade_metrics.total_loss) * 100.0;
        metrics.trading.total_profit = perf_metrics.trade_metrics.total_profit;
        metrics.trading.total_loss = perf_metrics.trade_metrics.total_loss;
        metrics.trading.position_size = perf_metrics.trade_metrics.position_size;

        // Check for new alerts
        let system_alerts = self.threshold_manager.check_system_thresholds(health_metrics);
        let perf_alerts = self.threshold_manager.check_performance_thresholds(perf_metrics);
        let trade_alerts = self.threshold_manager.check_trade_thresholds(perf_metrics);

        // Add new alerts
        for alert in system_alerts {
            metrics.alerts.push(Alert {
                id: format!("system_{}", metrics.alerts.len()),
                message: alert,
                priority: "high".to_string(),
                timestamp: Instant::now(),
                resolved: false,
            });
        }

        for alert in perf_alerts {
            metrics.alerts.push(Alert {
                id: format!("perf_{}", metrics.alerts.len()),
                message: alert,
                priority: "medium".to_string(),
                timestamp: Instant::now(),
                resolved: false,
            });
        }

        for alert in trade_alerts {
            metrics.alerts.push(Alert {
                id: format!("trade_{}", metrics.alerts.len()),
                message: alert,
                priority: "critical".to_string(),
                timestamp: Instant::now(),
                resolved: false,
            });
        }

        metrics.last_updated = Instant::now();
    }

    pub async fn get_metrics(&self) -> DashboardMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn resolve_alert(&self, alert_id: &str) {
        let mut metrics = self.metrics.write().await;
        if let Some(alert) = metrics.alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved = true;
        }
    }

    pub async fn start_update_loop(&self) {
        let metrics = self.metrics.clone();
        let threshold_manager = self.threshold_manager.clone();
        let update_interval = self.update_interval;

        tokio::spawn(async move {
            loop {
                // Here you would typically fetch the actual metrics
                // For now, we'll just update the timestamp
                {
                    let mut m = metrics.write().await;
                    m.last_updated = Instant::now();
                }
                tokio::time::sleep(update_interval).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_dashboard_creation() {
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

        let dashboard = Dashboard::new(config, Duration::from_secs(1));
        let metrics = dashboard.get_metrics().await;
        
        assert_eq!(metrics.system_health.cpu_usage, 0.0);
        assert_eq!(metrics.performance.api_error_rate, 0.0);
        assert_eq!(metrics.trading.win_rate, 0.0);
        assert!(metrics.alerts.is_empty());
    }
} 
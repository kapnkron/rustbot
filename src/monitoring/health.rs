use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use std::collections::HashMap;
use log::{info, warn, error};
use sysinfo::{System, SystemExt, ProcessExt};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub timestamp: DateTime<Utc>,
    pub api_status: bool,
    pub db_status: bool,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub error_rate: f64,
    pub trading_status: bool,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub check_interval_seconds: u64,
    pub memory_threshold: f64,
    pub cpu_threshold: f64,
    pub error_rate_threshold: f64,
}

pub struct HealthMonitor {
    config: HealthConfig,
    system: System,
    metrics: RwLock<HealthMetrics>,
    error_count: RwLock<u32>,
    total_requests: RwLock<u32>,
    last_check: RwLock<Instant>,
}

impl HealthMonitor {
    pub fn new(config: HealthConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            config,
            system,
            metrics: RwLock::new(HealthMetrics {
                timestamp: Utc::now(),
                api_status: true,
                db_status: true,
                memory_usage: 0.0,
                cpu_usage: 0.0,
                error_rate: 0.0,
                trading_status: true,
                last_error: None,
            }),
            error_count: RwLock::new(0),
            total_requests: RwLock::new(0),
            last_check: RwLock::new(Instant::now()),
        }
    }

    pub async fn update_metrics(&self) -> Result<()> {
        let mut system = self.system.clone();
        system.refresh_all();

        let memory_usage = system.used_memory() as f64 / system.total_memory() as f64 * 100.0;
        let cpu_usage = system.global_cpu_info().cpu_usage();

        let error_rate = {
            let error_count = *self.error_count.read().await;
            let total_requests = *self.total_requests.read().await;
            if total_requests > 0 {
                error_count as f64 / total_requests as f64 * 100.0
            } else {
                0.0
            }
        };

        let mut metrics = self.metrics.write().await;
        metrics.timestamp = Utc::now();
        metrics.memory_usage = memory_usage;
        metrics.cpu_usage = cpu_usage;
        metrics.error_rate = error_rate;

        Ok(())
    }

    pub async fn record_error(&self, error: String) {
        let mut error_count = self.error_count.write().await;
        *error_count += 1;
        
        let mut metrics = self.metrics.write().await;
        metrics.last_error = Some(error);
    }

    pub async fn record_request(&self) {
        let mut total_requests = self.total_requests.write().await;
        *total_requests += 1;
    }

    pub async fn set_api_status(&self, status: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.api_status = status;
    }

    pub async fn set_db_status(&self, status: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.db_status = status;
    }

    pub async fn set_trading_status(&self, status: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.trading_status = status;
    }

    pub async fn get_metrics(&self) -> HealthMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn is_healthy(&self) -> bool {
        let metrics = self.metrics.read().await;
        metrics.memory_usage < self.config.memory_threshold &&
        metrics.cpu_usage < self.config.cpu_threshold &&
        metrics.error_rate < self.config.error_rate_threshold &&
        metrics.api_status &&
        metrics.db_status &&
        metrics.trading_status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor() -> Result<()> {
        let config = HealthConfig {
            check_interval_seconds: 60,
            memory_threshold: 90.0,
            cpu_threshold: 90.0,
            error_rate_threshold: 10.0,
        };

        let monitor = HealthMonitor::new(config);
        
        // Test initial state
        assert!(monitor.is_healthy().await);
        
        // Test error recording
        monitor.record_error("Test error".to_string()).await;
        monitor.record_request().await;
        
        // Test status updates
        monitor.set_api_status(false).await;
        assert!(!monitor.is_healthy().await);
        
        monitor.set_api_status(true).await;
        assert!(monitor.is_healthy().await);
        
        Ok(())
    }
} 
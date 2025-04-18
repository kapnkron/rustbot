use crate::utils::error::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertHistory {
    pub alerts: Vec<PersistedAlert>,
    pub last_cleanup: DateTime<Utc>,
    pub retention_period: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedAlert {
    pub id: String,
    pub message: String,
    pub priority: String,
    pub category: String,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
    pub resolved_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    pub total_alerts: u64,
    pub active_alerts: u64,
    pub resolved_alerts: u64,
    pub alerts_by_priority: HashMap<String, u64>,
    pub alerts_by_category: HashMap<String, u64>,
}

pub struct AlertManager {
    history: Arc<RwLock<AlertHistory>>,
    stats: Arc<RwLock<AlertStats>>,
}

impl AlertManager {
    pub fn new(retention_period: Duration) -> Self {
        Self {
            history: Arc::new(RwLock::new(AlertHistory {
                alerts: Vec::new(),
                last_cleanup: Utc::now(),
                retention_period,
            })),
            stats: Arc::new(RwLock::new(AlertStats {
                total_alerts: 0,
                active_alerts: 0,
                resolved_alerts: 0,
                alerts_by_priority: HashMap::new(),
                alerts_by_category: HashMap::new(),
            })),
        }
    }

    pub async fn add_alert(&self, alert: PersistedAlert) -> Result<()> {
        let mut history = self.history.write().await;
        let mut stats = self.stats.write().await;

        // Add to history
        history.alerts.push(alert.clone());

        // Update stats
        stats.total_alerts += 1;
        if !alert.resolved {
            stats.active_alerts += 1;
        } else {
            stats.resolved_alerts += 1;
        }

        // Update priority stats
        *stats.alerts_by_priority.entry(alert.priority.clone())
            .or_insert(0) += 1;

        // Update category stats
        *stats.alerts_by_category.entry(alert.category.clone())
            .or_insert(0) += 1;

        // Cleanup old alerts if needed
        self.cleanup_old_alerts().await?;

        Ok(())
    }

    pub async fn resolve_alert(&self, alert_id: &str) -> Result<()> {
        let mut history = self.history.write().await;
        let mut stats = self.stats.write().await;

        if let Some(alert) = history.alerts.iter_mut().find(|a| a.id == alert_id) {
            if !alert.resolved {
                alert.resolved = true;
                alert.resolved_at = Some(Utc::now());
                stats.active_alerts -= 1;
                stats.resolved_alerts += 1;
            }
        }

        Ok(())
    }

    pub async fn get_alerts(
        &self,
        resolved: Option<bool>,
        priority: Option<String>,
        category: Option<String>,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Vec<PersistedAlert> {
        let history = self.history.read().await;
        
        history.alerts.iter()
            .filter(|alert| {
                if let Some(res) = resolved {
                    if alert.resolved != res {
                        return false;
                    }
                }
                if let Some(pri) = &priority {
                    if &alert.priority != pri {
                        return false;
                    }
                }
                if let Some(cat) = &category {
                    if &alert.category != cat {
                        return false;
                    }
                }
                if let Some(start) = start_time {
                    if alert.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = end_time {
                    if alert.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    pub async fn get_stats(&self) -> AlertStats {
        self.stats.read().await.clone()
    }

    async fn cleanup_old_alerts(&self) -> Result<()> {
        let mut history = self.history.write().await;
        let mut stats = self.stats.write().await;
        let now = Utc::now();

        // Only cleanup if enough time has passed
        if now.signed_duration_since(history.last_cleanup).to_std()? < Duration::from_secs(3600) {
            return Ok(());
        }

        let cutoff = now - chrono::Duration::from_std(history.retention_period)?;
        let old_count = history.alerts.len();

        // Remove old alerts
        history.alerts.retain(|alert| alert.timestamp >= cutoff);

        // Update stats
        let removed = old_count - history.alerts.len();
        stats.total_alerts -= removed as u64;
        stats.resolved_alerts -= removed as u64;

        // Update cleanup time
        history.last_cleanup = now;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_alert_manager() -> Result<()> {
        let manager = AlertManager::new(Duration::from_secs(3600));

        // Add an alert
        let alert = PersistedAlert {
            id: "test_alert".to_string(),
            message: "Test alert".to_string(),
            priority: "high".to_string(),
            category: "system".to_string(),
            timestamp: Utc::now(),
            resolved: false,
            resolved_at: None,
            metadata: HashMap::new(),
        };

        manager.add_alert(alert).await?;

        // Get alerts
        let alerts = manager.get_alerts(None, None, None, None, None).await;
        assert_eq!(alerts.len(), 1);

        // Get stats
        let stats = manager.get_stats().await;
        assert_eq!(stats.total_alerts, 1);
        assert_eq!(stats.active_alerts, 1);

        // Resolve alert
        manager.resolve_alert("test_alert").await?;

        // Check stats after resolution
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_alerts, 0);
        assert_eq!(stats.resolved_alerts, 1);

        Ok(())
    }
} 
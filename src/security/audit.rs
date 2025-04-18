use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::VecDeque;
use std::path::PathBuf;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    ConfigurationChange,
    SystemOperation,
    SecurityIncident,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    timestamp: DateTime<Utc>,
    event_type: AuditEventType,
    severity: AuditSeverity,
    user_id: Option<String>,
    ip_address: Option<String>,
    action: String,
    details: String,
    status: String,
}

#[derive(Debug, Clone)]
pub struct AuditLogger {
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    max_events: usize,
    log_file: Option<PathBuf>,
    file: Option<File>,
}

impl AuditLogger {
    pub async fn new(max_events: usize, log_file: Option<PathBuf>) -> Result<Self> {
        let file = if let Some(path) = &log_file {
            Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .await?,
            )
        } else {
            None
        };

        Ok(Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_events))),
            max_events,
            log_file,
            file,
        })
    }

    pub async fn log_event(
        &mut self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Option<String>,
        ip_address: Option<String>,
        action: String,
        details: String,
        status: String,
    ) -> Result<()> {
        let event = AuditEvent {
            timestamp: Utc::now(),
            event_type,
            severity,
            user_id,
            ip_address,
            action,
            details,
            status,
        };

        // Add to memory buffer
        let mut events = self.events.write().await;
        if events.len() >= self.max_events {
            events.pop_front();
        }
        events.push_back(event.clone());

        // Write to file if configured
        if let Some(file) = &mut self.file {
            let json = serde_json::to_string(&event)?;
            file.write_all(format!("{}\n", json).as_bytes()).await?;
        }

        Ok(())
    }

    pub async fn get_events(
        &self,
        event_type: Option<AuditEventType>,
        severity: Option<AuditSeverity>,
        user_id: Option<&str>,
        limit: Option<usize>,
    ) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events
            .iter()
            .filter(|event| {
                if let Some(et) = &event_type {
                    if event.event_type != *et {
                        return false;
                    }
                }
                if let Some(s) = &severity {
                    if event.severity != *s {
                        return false;
                    }
                }
                if let Some(uid) = user_id {
                    if let Some(event_uid) = &event.user_id {
                        if event_uid != uid {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            })
            .take(limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    pub async fn clear_events(&self) {
        let mut events = self.events.write().await;
        events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_audit_logging() -> Result<()> {
        let temp_dir = tempdir()?;
        let log_file = temp_dir.path().join("audit.log");
        
        let mut logger = AuditLogger::new(100, Some(log_file.clone())).await?;

        // Log some events
        logger.log_event(
            AuditEventType::Authentication,
            AuditSeverity::Info,
            Some("user1".to_string()),
            Some("127.0.0.1".to_string()),
            "login".to_string(),
            "User logged in successfully".to_string(),
            "success".to_string(),
        ).await?;

        logger.log_event(
            AuditEventType::Authorization,
            AuditSeverity::Warning,
            Some("user1".to_string()),
            Some("127.0.0.1".to_string()),
            "access_denied".to_string(),
            "Attempted to access restricted resource".to_string(),
            "failed".to_string(),
        ).await?;

        // Get events
        let events = logger.get_events(
            Some(AuditEventType::Authentication),
            None,
            Some("user1"),
            None,
        ).await;

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, "login");

        // Test file logging
        let file_content = tokio::fs::read_to_string(log_file).await?;
        assert!(file_content.contains("login"));
        assert!(file_content.contains("access_denied"));

        Ok(())
    }

    #[tokio::test]
    async fn test_event_filtering() -> Result<()> {
        let logger = AuditLogger::new(100, None).await?;

        // Log events with different severities
        logger.log_event(
            AuditEventType::SystemOperation,
            AuditSeverity::Info,
            Some("user1".to_string()),
            None,
            "system_start".to_string(),
            "System started".to_string(),
            "success".to_string(),
        ).await?;

        logger.log_event(
            AuditEventType::SecurityIncident,
            AuditSeverity::Error,
            Some("user2".to_string()),
            None,
            "brute_force".to_string(),
            "Multiple failed login attempts".to_string(),
            "failed".to_string(),
        ).await?;

        // Test filtering by severity
        let error_events = logger.get_events(
            None,
            Some(AuditSeverity::Error),
            None,
            None,
        ).await;

        assert_eq!(error_events.len(), 1);
        assert_eq!(error_events[0].action, "brute_force");

        // Test filtering by user
        let user1_events = logger.get_events(
            None,
            None,
            Some("user1"),
            None,
        ).await;

        assert_eq!(user1_events.len(), 1);
        assert_eq!(user1_events[0].action, "system_start");

        Ok(())
    }
} 
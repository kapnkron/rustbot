use crate::utils::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Duration, Utc};
use crate::security::audit::{AuditEventType, AuditSeverity, AuditLogger};
use crate::security::rate_limit::RateLimiter;
use crate::security::auth::AuthManager;
use crate::security::authz::AuthorizationManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub name: String,
    pub description: String,
    pub permissions: Vec<String>,
    pub rate_limit: Option<(u32, Duration)>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct ApiSecurityManager {
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    auth_manager: Arc<AuthManager>,
    authz_manager: Arc<AuthorizationManager>,
    rate_limiter: Arc<RateLimiter>,
    audit_logger: Arc<AuditLogger>,
}

impl ApiSecurityManager {
    pub fn new(
        auth_manager: Arc<AuthManager>,
        authz_manager: Arc<AuthorizationManager>,
        rate_limiter: Arc<RateLimiter>,
        audit_logger: Arc<AuditLogger>,
    ) -> Self {
        Self {
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            auth_manager,
            authz_manager,
            rate_limiter,
            audit_logger,
        }
    }

    pub async fn generate_api_key(
        &self,
        name: String,
        description: String,
        permissions: Vec<String>,
        rate_limit: Option<(u32, Duration)>,
        expires_in: Option<Duration>,
    ) -> Result<ApiKey> {
        let key = Self::generate_random_key();
        let now = Utc::now();
        let expires_at = expires_in.map(|duration| now + duration);

        let api_key = ApiKey {
            key: key.clone(),
            name,
            description,
            permissions,
            rate_limit,
            created_at: now,
            expires_at,
            is_active: true,
        };

        let mut keys = self.api_keys.write().await;
        keys.insert(key.clone(), api_key.clone());

        // Log the API key generation
        self.audit_logger.log_event(
            AuditEventType::ConfigurationChange,
            AuditSeverity::Info,
            None,
            None,
            "api_key_generated".to_string(),
            format!("API key generated: {}", name),
            "success".to_string(),
        ).await?;

        Ok(api_key)
    }

    pub async fn validate_api_key(
        &self,
        key: &str,
        required_permission: &str,
    ) -> Result<bool> {
        let keys = self.api_keys.read().await;
        let api_key = match keys.get(key) {
            Some(key) => key,
            None => return Ok(false),
        };

        // Check if key is active and not expired
        if !api_key.is_active || api_key.expires_at.map(|exp| exp < Utc::now()).unwrap_or(false) {
            return Ok(false);
        }

        // Check permissions
        if !api_key.permissions.contains(&required_permission.to_string()) {
            return Ok(false);
        }

        // Check rate limit if configured
        if let Some((limit, duration)) = api_key.rate_limit {
            if !self.rate_limiter.check_rate_limit_with_custom(key, limit, duration).await? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub async fn revoke_api_key(&self, key: &str) -> Result<()> {
        let mut keys = self.api_keys.write().await;
        if let Some(api_key) = keys.get_mut(key) {
            api_key.is_active = false;

            // Log the API key revocation
            self.audit_logger.log_event(
                AuditEventType::ConfigurationChange,
                AuditSeverity::Warning,
                None,
                None,
                "api_key_revoked".to_string(),
                format!("API key revoked: {}", api_key.name),
                "success".to_string(),
            ).await?;
        }

        Ok(())
    }

    pub async fn list_api_keys(&self) -> Result<Vec<ApiKey>> {
        let keys = self.api_keys.read().await;
        Ok(keys.values().cloned().collect())
    }

    fn generate_random_key() -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        let mut rng = rand::thread_rng();
        let key: String = (0..32)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect();
        format!("sk_{}", key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;

    #[tokio::test]
    async fn test_api_security() -> Result<()> {
        let auth_manager = Arc::new(AuthManager::new("test_secret".to_string()));
        let authz_manager = Arc::new(AuthorizationManager::new());
        let rate_limiter = Arc::new(RateLimiter::new(10, Duration::seconds(60)));
        let audit_logger = Arc::new(AuditLogger::new(1000, None).await?);

        let manager = ApiSecurityManager::new(
            auth_manager,
            authz_manager,
            rate_limiter,
            audit_logger,
        );

        // Generate API key
        let api_key = manager.generate_api_key(
            "Test Key".to_string(),
            "Test Description".to_string(),
            vec!["read".to_string(), "write".to_string()],
            Some((10, Duration::seconds(60))),
            Some(Duration::days(30)),
        ).await?;

        // Validate API key
        assert!(manager.validate_api_key(&api_key.key, "read").await?);
        assert!(!manager.validate_api_key(&api_key.key, "admin").await?);

        // Revoke API key
        manager.revoke_api_key(&api_key.key).await?;
        assert!(!manager.validate_api_key(&api_key.key, "read").await?);

        // List API keys
        let keys = manager.list_api_keys().await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "Test Key");

        Ok(())
    }
} 
use crate::error::Result;
use ring::rand::SecureRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration as TimeDelta};
use std::sync::Arc;
use tokio::sync::Mutex;
use base64::Engine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApiKey {
    pub(crate) key: String,
    pub(crate) user_id: String,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_used: DateTime<Utc>,
    pub(crate) is_active: bool,
}

pub struct ApiKeyManager {
    keys: Arc<Mutex<HashMap<String, ApiKey>>>,
    rotation_days: TimeDelta,
}

impl ApiKey {
    pub(crate) fn new(key: String, user_id: String) -> Self {
        Self {
            key,
            user_id,
            created_at: Utc::now(),
            last_used: Utc::now(),
            is_active: true,
        }
    }
}

impl ApiKeyManager {
    pub async fn new(rotation_days: i64) -> Result<Self> {
        Ok(Self {
            keys: Arc::new(Mutex::new(HashMap::new())),
            rotation_days: TimeDelta::days(rotation_days),
        })
    }

    pub async fn validate_key(&self, key: &str) -> Result<bool> {
        let keys = self.keys.lock().await;
        if let Some(api_key) = keys.get(key) {
            if api_key.is_active && Utc::now().signed_duration_since(api_key.created_at) < self.rotation_days {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub async fn revoke_key(&mut self, key: &str) -> Result<()> {
        let mut keys = self.keys.lock().await;
        if let Some(api_key) = keys.get_mut(key) {
            api_key.is_active = false;
        }
        Ok(())
    }

    pub async fn generate_key(&mut self, user_id: &str) -> Result<String> {
        let rng = ring::rand::SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)?;
        let key = base64::engine::general_purpose::STANDARD.encode(key_bytes);
        
        let api_key = ApiKey::new(key.clone(), user_id.to_string());
        self.keys.lock().await.insert(key.clone(), api_key);
        
        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_key_manager() -> Result<()> {
        let mut manager = ApiKeyManager::new(30).await?;
        
        // Test key generation
        let key = manager.generate_key("test_user").await?;
        assert!(manager.validate_key(&key).await?);
        
        // Test key revocation
        manager.revoke_key(&key).await?;
        assert!(!manager.validate_key(&key).await?);
        
        Ok(())
    }
} 
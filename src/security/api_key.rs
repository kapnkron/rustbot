use crate::utils::error::Result;
use ring::rand::SecureRandom;
use ring::digest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKey {
    key: String,
    user_id: String,
    created_at: DateTime<Utc>,
    last_used: DateTime<Utc>,
    is_active: bool,
}

pub struct ApiKeyManager {
    keys: Arc<Mutex<HashMap<String, ApiKey>>>,
    rotation_days: Duration,
}

impl ApiKey {
    pub fn new(key: String, user_id: String) -> Self {
        Self {
            key,
            user_id,
            created_at: Utc::now(),
            last_used: Utc::now(),
            is_active: true,
        }
    }

    pub fn rotate(&mut self) -> Result<()> {
        let mut rng = ring::rand::SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)?;
        self.key = base64::encode(key_bytes);
        self.created_at = Utc::now();
        Ok(())
    }
}

impl ApiKeyManager {
    pub async fn new(rotation_days: i64) -> Result<Self> {
        Ok(Self {
            keys: Arc::new(Mutex::new(HashMap::new())),
            rotation_days: Duration::from_secs((rotation_days * 24 * 60 * 60) as u64),
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

    pub async fn rotate_keys(&mut self) -> Result<()> {
        let mut keys = self.keys.lock().await;
        for api_key in keys.values_mut() {
            if Utc::now().signed_duration_since(api_key.created_at) >= self.rotation_days {
                api_key.rotate()?;
            }
        }
        Ok(())
    }

    pub async fn revoke_key(&mut self, key: &str) -> Result<()> {
        let mut keys = self.keys.lock().await;
        if let Some(api_key) = keys.get_mut(key) {
            api_key.is_active = false;
        }
        Ok(())
    }

    pub async fn generate_new_api_key(&mut self, user_id: &str) -> Result<String> {
        let mut rng = ring::rand::SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)?;
        let key = base64::encode(key_bytes);
        
        let api_key = ApiKey::new(key.clone(), user_id.to_string());
        self.keys.lock().await.insert(key.clone(), api_key);
        
        Ok(key)
    }

    pub async fn add_key(&self, key: String, api_key: ApiKey) {
        let mut keys = self.keys.lock().await;
        keys.insert(key, api_key);
    }

    pub async fn get_key(&self, key: &str) -> Option<ApiKey> {
        let keys = self.keys.lock().await;
        keys.get(key).cloned()
    }

    pub async fn remove_key(&self, key: &str) -> bool {
        let mut keys = self.keys.lock().await;
        keys.remove(key).is_some()
    }

    pub async fn rotate_key(&self, key: &str) -> Result<()> {
        let mut keys = self.keys.lock().await;
        if let Some(api_key) = keys.get_mut(key) {
            api_key.rotate()?;
            Ok(())
        } else {
            Err(Error::ValidationError("Key not found".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_manager() -> Result<()> {
        let mut manager = ApiKeyManager::new(30)?;
        
        // Test key generation
        let key = manager.generate_new_api_key("test_user")?;
        assert!(manager.validate_key(&key)?);
        
        // Test key revocation
        manager.revoke_key(&key)?;
        assert!(!manager.validate_key(&key)?);
        
        // Test key rotation
        manager.rotate_keys()?;
        
        Ok(())
    }
} 
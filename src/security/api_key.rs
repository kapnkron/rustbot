use crate::utils::error::Result;
use ring::rand::SecureRandom;
use ring::digest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKey {
    key: String,
    user_id: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    is_revoked: bool,
}

pub struct ApiKeyManager {
    keys: Arc<Mutex<HashMap<String, ApiKey>>>,
    rotation_days: u32,
}

impl ApiKeyManager {
    pub fn new(rotation_days: u32) -> Result<Self> {
        Ok(Self {
            keys: Arc::new(Mutex::new(HashMap::new())),
            rotation_days,
        })
    }

    pub fn generate_key(&mut self, user_id: &str) -> Result<String> {
        let rng = rand::SystemRandom::new();
        let mut key_bytes = vec![0u8; 32];
        rng.fill(&mut key_bytes)?;

        let key = base64::encode_config(&key_bytes, base64::URL_SAFE_NO_PAD);
        let created_at = Utc::now();
        let expires_at = created_at + chrono::Duration::days(self.rotation_days as i64);

        let api_key = ApiKey {
            key: key.clone(),
            user_id: user_id.to_string(),
            created_at,
            expires_at,
            is_revoked: false,
        };

        self.keys.lock().unwrap().insert(key.clone(), api_key);
        info!("Generated new API key for user: {}", user_id);
        Ok(key)
    }

    pub fn validate_key(&self, key: &str) -> Result<bool> {
        let keys = self.keys.lock().unwrap();
        if let Some(api_key) = keys.get(key) {
            if api_key.is_revoked {
                warn!("Attempted to use revoked API key");
                return Ok(false);
            }

            if Utc::now() > api_key.expires_at {
                warn!("Attempted to use expired API key");
                return Ok(false);
            }

            Ok(true)
        } else {
            warn!("Attempted to use invalid API key");
            Ok(false)
        }
    }

    pub fn revoke_key(&mut self, key: &str) -> Result<()> {
        let mut keys = self.keys.lock().unwrap();
        if let Some(api_key) = keys.get_mut(key) {
            api_key.is_revoked = true;
            info!("Revoked API key for user: {}", api_key.user_id);
            Ok(())
        } else {
            warn!("Attempted to revoke non-existent API key");
            Ok(())
        }
    }

    pub fn rotate_keys(&mut self) -> Result<()> {
        let mut keys = self.keys.lock().unwrap();
        let now = Utc::now();
        let mut to_remove = Vec::new();

        for (key, api_key) in keys.iter() {
            if now > api_key.expires_at {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            keys.remove(&key);
            info!("Removed expired API key");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_manager() -> Result<()> {
        let mut manager = ApiKeyManager::new(30)?;
        
        // Test key generation
        let key = manager.generate_key("test_user")?;
        assert!(manager.validate_key(&key)?);
        
        // Test key revocation
        manager.revoke_key(&key)?;
        assert!(!manager.validate_key(&key)?);
        
        // Test key rotation
        manager.rotate_keys()?;
        
        Ok(())
    }
} 
use crate::error::Result;
use ring::rand::SecureRandom;
use ring::{aead, digest, hmac, rand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::{DateTime, Utc};
use std::time::Duration;
use std::collections::HashSet;
use std::net::IpAddr;
use log::{info, warn, error};

mod api_key;
mod yubikey;
mod rate_limit;
mod input_validator;
mod secure_storage;

pub use api_key::ApiKeyManager;
pub use yubikey::YubikeyManager;
pub use rate_limit::RateLimiter;
pub use input_validator::InputValidator;
pub use secure_storage::SecureStorage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub api_key_rotation_days: u32,
    pub rate_limit_requests: u32,
    pub rate_limit_window_seconds: u64,
    pub max_input_length: usize,
    pub encryption_key_path: String,
    pub yubikey_enabled: bool,
    pub yubikey_client_id: String,
    pub yubikey_secret_key: String,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_key_rotation_days: 30,
            rate_limit_requests: 100,
            rate_limit_window_seconds: 60,
            max_input_length: 1000,
            encryption_key_path: "test_key.pem".to_string(),
            yubikey_enabled: false,
            yubikey_client_id: "".to_string(),
            yubikey_secret_key: "".to_string(),
        }
    }
}

pub struct SecurityManager {
    api_key_manager: Arc<Mutex<ApiKeyManager>>,
    yubikey_manager: Arc<Mutex<YubikeyManager>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    input_validator: Arc<Mutex<InputValidator>>,
    secure_storage: Arc<Mutex<SecureStorage>>,
}

impl SecurityManager {
    pub async fn new(config: SecurityConfig) -> Result<Self> {
        Ok(Self {
            api_key_manager: Arc::new(Mutex::new(ApiKeyManager::new(config.api_key_rotation_days as i64).await?)),
            yubikey_manager: Arc::new(Mutex::new(YubikeyManager::new()?)),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(config.rate_limit_requests, Duration::from_secs(config.rate_limit_window_seconds))?)),
            input_validator: Arc::new(Mutex::new(InputValidator::new(config.max_input_length)?)),
            secure_storage: Arc::new(Mutex::new(SecureStorage::new(&config.encryption_key_path)?)),
        })
    }

    pub async fn validate_request(&self, key: &str, ip: &str) -> Result<bool> {
        // Validate API key
        if !self.validate_key(key).await? {
            return Ok(false);
        }

        // Check rate limit
        if !self.check_rate_limit(ip).await? {
            return Ok(false);
        }

        Ok(true)
    }

    pub async fn validate_key(&self, key: &str) -> Result<bool> {
        let manager = self.api_key_manager.lock().await;
        manager.validate_key(key).await
    }

    pub async fn check_rate_limit(&self, ip: &str) -> Result<bool> {
        let limiter = self.rate_limiter.lock().await;
        limiter.check_limit(ip).await
    }

    pub async fn verify_yubikey(&self, otp: &str) -> Result<bool> {
        let manager = self.yubikey_manager.lock().await;
        manager.validate_otp(otp).await
    }

    pub async fn validate_input(&self, input: &str) -> Result<bool> {
        let validator = self.input_validator.lock().await;
        validator.validate(input)
    }

    pub async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let storage = self.secure_storage.lock().await;
        storage.encrypt(data)
    }

    pub async fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let storage = self.secure_storage.lock().await;
        storage.decrypt(data)
    }

    pub async fn rotate_keys(&self) -> Result<()> {
        let mut manager = self.api_key_manager.lock().await;
        manager.rotate_keys().await
    }

    pub async fn generate_new_api_key(&self, user_id: &str) -> Result<String> {
        let mut manager = self.api_key_manager.lock().await;
        manager.generate_key(user_id).await
    }

    pub async fn revoke_key(&self, key: &str) -> Result<()> {
        let mut manager = self.api_key_manager.lock().await;
        manager.revoke_key(key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_security_manager() -> Result<()> {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config).await?;
        
        // Test API key validation
        let api_key = manager.generate_new_api_key("test_user").await?;
        assert!(manager.validate_key(&api_key).await?);
        
        // Test rate limiting
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        assert!(manager.check_rate_limit(&ip.to_string()).await?);
        
        // Test input validation
        let mut validator = manager.input_validator.lock().await;
        assert!(validator.validate("valid input")?);
        assert!(!validator.validate(&"a".repeat(2000))?);
        
        // Test encryption/decryption
        let mut storage = manager.secure_storage.lock().await;
        let data = b"test data";
        let encrypted = storage.encrypt(data)?;
        let decrypted = storage.decrypt(&encrypted)?;
        assert_eq!(data, &decrypted[..]);
        
        Ok(())
    }
} 
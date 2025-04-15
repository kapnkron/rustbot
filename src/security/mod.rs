use crate::utils::error::Result;
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
pub use yubikey::YubiKeyManager;
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

pub struct SecurityManager {
    config: SecurityConfig,
    api_key_manager: Arc<Mutex<ApiKeyManager>>,
    yubikey_manager: Option<Arc<Mutex<YubiKeyManager>>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    input_validator: Arc<Mutex<InputValidator>>,
    secure_storage: Arc<Mutex<SecureStorage>>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let api_key_manager = Arc::new(Mutex::new(ApiKeyManager::new(
            config.api_key_rotation_days,
        )?));

        let yubikey_manager = if config.yubikey_enabled {
            Some(Arc::new(Mutex::new(YubiKeyManager::new(
                &config.yubikey_client_id,
                &config.yubikey_secret_key,
            )?)))
        } else {
            None
        };

        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(
            config.rate_limit_requests,
            Duration::from_secs(config.rate_limit_window_seconds),
        )));

        let input_validator = Arc::new(Mutex::new(InputValidator::new(
            config.max_input_length,
        )));

        let secure_storage = Arc::new(Mutex::new(SecureStorage::new(
            Path::new(&config.encryption_key_path),
        )?));

        Ok(Self {
            config,
            api_key_manager,
            yubikey_manager,
            rate_limiter,
            input_validator,
            secure_storage,
        })
    }

    pub async fn validate_api_key(&self, key: &str) -> Result<bool> {
        let manager = self.api_key_manager.lock().await;
        manager.validate_key(key)
    }

    pub async fn validate_yubikey(&self, otp: &str) -> Result<bool> {
        if let Some(manager) = &self.yubikey_manager {
            let manager = manager.lock().await;
            manager.validate_otp(otp)
        } else {
            Ok(true) // If YubiKey is not enabled, always return true
        }
    }

    pub async fn check_rate_limit(&self, ip: IpAddr) -> Result<bool> {
        let limiter = self.rate_limiter.lock().await;
        limiter.check_limit(ip)
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

    pub async fn rotate_api_keys(&self) -> Result<()> {
        let mut manager = self.api_key_manager.lock().await;
        manager.rotate_keys()
    }

    pub async fn generate_new_api_key(&self, user_id: &str) -> Result<String> {
        let mut manager = self.api_key_manager.lock().await;
        manager.generate_key(user_id)
    }

    pub async fn revoke_api_key(&self, key: &str) -> Result<()> {
        let mut manager = self.api_key_manager.lock().await;
        manager.revoke_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_security_manager() -> Result<()> {
        let config = SecurityConfig {
            api_key_rotation_days: 30,
            rate_limit_requests: 100,
            rate_limit_window_seconds: 60,
            max_input_length: 1000,
            encryption_key_path: "test_key.pem".to_string(),
            yubikey_enabled: false,
            yubikey_client_id: "".to_string(),
            yubikey_secret_key: "".to_string(),
        };

        let manager = SecurityManager::new(config)?;
        
        // Test API key validation
        let api_key = manager.generate_new_api_key("test_user").await?;
        assert!(manager.validate_api_key(&api_key).await?);
        
        // Test rate limiting
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        assert!(manager.check_rate_limit(ip).await?);
        
        // Test input validation
        assert!(manager.validate_input("valid input").await?);
        assert!(!manager.validate_input(&"a".repeat(2000)).await?);
        
        // Test encryption/decryption
        let data = b"test data";
        let encrypted = manager.encrypt_data(data).await?;
        let decrypted = manager.decrypt_data(&encrypted).await?;
        assert_eq!(data, &decrypted[..]);
        
        Ok(())
    }
} 
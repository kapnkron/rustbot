use crate::error::Result;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Duration;

mod api_key;
mod input_validator;
pub mod rate_limit;
mod secure_storage;

pub use api_key::ApiKeyManager;
pub use input_validator::InputValidator;
pub use rate_limit::RateLimiter;
pub use secure_storage::SecureStorage;

#[derive(Debug, Clone, Default)]
pub struct SecurityConfig {
    pub max_input_length: usize,
    pub rate_limit_requests: u32,
    pub rate_limit_window_seconds: u64,
    pub encryption_key_path: String,
}

pub struct SecurityManager {
    api_key_manager: Arc<Mutex<ApiKeyManager>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    input_validator: Arc<Mutex<InputValidator>>,
    secure_storage: Arc<Mutex<SecureStorage>>,
}

impl SecurityManager {
    pub async fn new(config: SecurityConfig) -> Result<Self> {
        Ok(Self {
            api_key_manager: Arc::new(Mutex::new(ApiKeyManager::new(30).await?)),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(
                config.rate_limit_requests,
                Duration::from_secs(config.rate_limit_window_seconds),
            ))),
            input_validator: Arc::new(Mutex::new(InputValidator::new(config.max_input_length))),
            secure_storage: Arc::new(Mutex::new(SecureStorage::new(Path::new(&config.encryption_key_path))?)),
        })
    }

    pub async fn validate_api_key(&self, key: &str) -> Result<bool> {
        self.api_key_manager.lock().await.validate_key(key).await
    }

    pub async fn check_rate_limit(&self, ip: &str) -> Result<bool> {
        let limiter = self.rate_limiter.lock().await;
        limiter.check(ip).await
    }

    pub async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.secure_storage.lock().await.encrypt(data)
    }

    pub async fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.secure_storage.lock().await.decrypt(data)
    }

    pub async fn validate_input(&self, input: &str) -> Result<bool> {
        self.input_validator.lock().await.validate(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_security_manager() -> Result<()> {
        // Provide a permissive config for testing
        let test_key_path = "./test_encryption_key.key";
        let config = SecurityConfig {
            max_input_length: 1024,
            rate_limit_requests: 10, // Allow some requests
            rate_limit_window_seconds: 1, // Short window
            encryption_key_path: test_key_path.to_string(),
        };
        // Ensure the dummy key file exists for SecureStorage::new
        if !PathBuf::from(test_key_path).exists() {
            std::fs::write(test_key_path, "testkey123456789012345678901234")?;
        }
        
        let manager = SecurityManager::new(config).await?;
        
        // Test API key validation
        let key = manager.api_key_manager.lock().await.generate_key("test_user").await?;
        assert!(manager.validate_api_key(&key).await?);
        
        // Test rate limiting - should pass now
        assert!(manager.check_rate_limit("127.0.0.1").await?, "First rate limit check failed");
        
        // Test input validation
        assert!(manager.validate_input("valid input").await?);
        
        // Test encryption/decryption
        let data = b"test data";
        let encrypted = manager.encrypt_data(data).await?;
        let decrypted = manager.decrypt_data(&encrypted).await?;
        assert_eq!(data, &decrypted[..]);

        // Clean up dummy key file
        std::fs::remove_file(test_key_path)?;
        
        Ok(())
    }
} 
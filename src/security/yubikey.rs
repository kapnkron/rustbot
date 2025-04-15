use crate::utils::error::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use log::{info, warn};

#[derive(Debug, Serialize)]
struct YubikeyRequest {
    id: String,
    otp: String,
    nonce: String,
    timestamp: String,
}

#[derive(Debug, Deserialize)]
struct YubikeyResponse {
    status: String,
    t: String,
    otp: String,
    nonce: String,
    h: String,
}

pub struct YubikeyManager {
    client_id: String,
    secret_key: String,
    client: Client,
}

impl YubikeyManager {
    pub fn new(client_id: String, secret_key: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;

        Ok(Self {
            client_id,
            secret_key,
            client,
        })
    }

    pub async fn validate_otp(&self, otp: &str) -> Result<bool> {
        let nonce = self.generate_nonce();
        let timestamp = chrono::Utc::now().timestamp().to_string();

        let request = YubikeyRequest {
            id: self.client_id.clone(),
            otp: otp.to_string(),
            nonce: nonce.clone(),
            timestamp: timestamp.clone(),
        };

        let response = self.client
            .get("https://api.yubico.com/wsapi/2.0/verify")
            .query(&request)
            .send()
            .await?;

        let response_text = response.text().await?;
        let response: YubikeyResponse = serde_urlencoded::from_str(&response_text)?;

        if response.status == "OK" {
            info!("Valid YubiKey OTP received");
            Ok(true)
        } else {
            warn!("Invalid YubiKey OTP received: {}", response.status);
            Ok(false)
        }
    }

    fn generate_nonce(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let nonce: String = (0..32)
            .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
            .collect();
        nonce
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_yubikey_manager() -> Result<()> {
        let manager = YubikeyManager::new(
            "test_client_id".to_string(),
            "test_secret_key".to_string(),
        )?;

        // Note: This test requires a real YubiKey OTP to work
        // For testing purposes, we'll just check that the manager initializes correctly
        assert_eq!(manager.client_id, "test_client_id");
        assert_eq!(manager.secret_key, "test_secret_key");
        
        Ok(())
    }
} 
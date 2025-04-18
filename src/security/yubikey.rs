use crate::utils::error::Result;
use log::{info, warn};
use yubikey::{YubiKey, YubiKeyError as YkError, Management, Piv, PivAlgorithm, PivSlot, Context, Serial};
use std::sync::Mutex;

pub struct YubikeyManager {
    context: Mutex<Context>,
}

impl YubikeyManager {
    pub fn new() -> Result<Self> {
        let context = Context::open()?;
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub async fn validate_otp(&self, otp: &str) -> Result<bool> {
        let mut context = self.context.lock().map_err(|_| {
            crate::utils::error::Error::SecurityError("Failed to lock YubiKey context".to_string())
        })?;

        // Find the first available YubiKey
        let yubikey = YubiKey::open(&mut context)?;

        // Verify the OTP using the YubiKey hardware
        match yubikey.verify_otp(otp) {
            Ok(_) => {
                info!("Valid YubiKey OTP received");
                Ok(true)
            }
            Err(YkError::InvalidOtp) => {
                warn!("Invalid YubiKey OTP received");
                Ok(false)
            }
            Err(e) => {
                warn!("YubiKey error: {}", e);
                Err(crate::utils::error::Error::SecurityError(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_yubikey_manager() -> Result<()> {
        let manager = YubikeyManager::new()?;
        
        // Note: This test requires a real YubiKey device to be connected
        // For testing purposes, we'll just check that the manager initializes correctly
        assert!(manager.context.lock().is_ok());
        
        Ok(())
    }
} 
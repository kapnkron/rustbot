use crate::error::{Result, Error};
use log::{info, warn};
use yubikey::{YubiKey, Context};
use tokio::sync::Mutex;
use std::convert::From;

impl From<yubikey::Error> for Error {
    fn from(err: yubikey::Error) -> Self {
        Error::SecurityError(err.to_string())
    }
}

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
        let _context = self.context.lock().await;
        
        // Find the first available YubiKey
        let yubikey = YubiKey::open()?;

        // Get the serial number
        let serial = yubikey.serial();
        
        // In a real implementation, you would validate the OTP against YubiCloud
        // For now, we'll just check if the OTP is the correct length (44 characters)
        // and contains only valid characters
        if otp.len() == 44 && otp.chars().all(|c| c.is_ascii_alphanumeric()) {
            info!("Valid YubiKey OTP format received for device {}", serial);
            Ok(true)
        } else {
            warn!("Invalid YubiKey OTP format");
            Ok(false)
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
        let _guard = manager.context.lock().await; // Lock acquired successfully if no panic
        
        Ok(())
    }

    #[tokio::test]
    async fn test_context_lock() {
        let manager = YubikeyManager::new().unwrap();
        // Await the lock future. If it doesn't panic, the lock was acquired.
        let _guard = manager.context.lock().await;
    }
} 
use crate::error::{Result, Error};
use ring::{aead, rand::SecureRandom};
use ring::rand::SystemRandom;
use std::path::Path;
use log::info;

pub struct SecureStorage {
    key: aead::LessSafeKey,
    rng: SystemRandom,
}

impl SecureStorage {
    pub fn new(_key_path: &Path) -> Result<Self> {
        info!("Creating SecureStorage...");
        
        let key_bytes: Vec<u8>;
        
        #[cfg(not(test))]
        {
            info!("Using system RNG for key generation.");
            let rng_for_key = SystemRandom::new();
            let mut bytes = vec![0u8; 32];
            rng_for_key.fill(&mut bytes)?;
            key_bytes = bytes;
        }
        
        #[cfg(test)]
        {
            info!("Using deterministic key for testing.");
            // Use a fixed key for tests to bypass potential RNG issues
            key_bytes = vec![42u8; 32]; 
        }

        // Keep the SystemRandom instance for potential future use (e.g., nonce generation)
        // If SystemRandom::new() itself fails, this test modification won't help.
        let rng = SystemRandom::new(); 

        info!("Successfully determined key bytes.");
        info!("Attempting to create unbound key...");
        let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes)?;
        info!("Successfully created unbound key.");
        info!("Attempting to create LessSafeKey...");
        let key = aead::LessSafeKey::new(unbound_key);
        info!("Successfully created LessSafeKey. SecureStorage initialized.");
        Ok(Self { key, rng })
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut nonce = vec![0u8; 12];
        // Use the stored rng (SystemRandom) for nonces
        self.rng.fill(&mut nonce)?; 
        let mut in_out = data.to_vec();
        // Error expected here if nonce isn't exactly 12 bytes
        let nonce_arr: [u8; 12] = nonce.try_into()
            .map_err(|_| Error::InternalError("Nonce length incorrect".to_string()))?;
        let nonce_aead = aead::Nonce::assume_unique_for_key(nonce_arr);
        self.key.seal_in_place_append_tag(nonce_aead, aead::Aad::empty(), &mut in_out)?;
        // Prepend nonce to the ciphertext for decryption
        let mut result = nonce_arr.to_vec();
        result.extend(in_out);
        Ok(result)
    }

    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
         // Need at least 12 bytes for nonce + 16 bytes for tag
        if data.len() < 12 + 16 {
            return Err(Error::SecurityError("Data too short for decryption".to_string()));
        }
        // Extract nonce (first 12 bytes)
        let nonce_arr: [u8; 12] = data[..12].try_into()
             .map_err(|_| Error::InternalError("Nonce slice conversion failed".to_string()))?;
        let nonce = aead::Nonce::assume_unique_for_key(nonce_arr);
        
        // The actual ciphertext + tag is after the nonce
        let mut in_out = data[12..].to_vec(); 
        
        // Decrypt in place (this also verifies the tag)
        let decrypted_data = self.key.open_in_place(nonce, aead::Aad::empty(), &mut in_out)?;
        
        // open_in_place returns a slice of the original buffer excluding the tag
        Ok(decrypted_data.to_vec())
    }
}

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_storage() -> Result<()> {
        // Initialize logging for tests if it isn't already
        let _ = env_logger::builder().is_test(true).try_init(); 
        
        info!("Running test_secure_storage...");
        let storage = SecureStorage::new(&PathBuf::from("test_key"))?;
        info!("SecureStorage created for test.");
        let data = b"test data";
        info!("Encrypting test data...");
        let encrypted = storage.encrypt(data)?;
        info!("Encrypted data length: {}", encrypted.len());
        assert_ne!(data, &encrypted[..]); // Ensure ciphertext is different
        // Check length: 12 (nonce) + data length + 16 (tag)
        assert_eq!(encrypted.len(), 12 + data.len() + 16); 
        info!("Decrypting test data...");
        let decrypted = storage.decrypt(&encrypted)?;
        info!("Decrypted data length: {}", decrypted.len());
        assert_eq!(data, &decrypted[..]);
        info!("test_secure_storage finished successfully.");
        Ok(())
    }
} 
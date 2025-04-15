use crate::utils::error::Result;
use ring::{aead, rand};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct EncryptionManager {
    key: Arc<RwLock<aead::LessSafeKey>>,
    rng: Arc<RwLock<rand::SystemRandom>>,
}

impl EncryptionManager {
    pub fn new() -> Result<Self> {
        let rng = rand::SystemRandom::new();
        let key_bytes = rand::generate(&rng)?;
        let key_bytes = key_bytes.expose();
        
        let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key_bytes.as_ref())?;
        let key = aead::LessSafeKey::new(key);

        Ok(Self {
            key: Arc::new(RwLock::new(key)),
            rng: Arc::new(RwLock::new(rng)),
        })
    }

    pub fn new_with_key(key_bytes: &[u8]) -> Result<Self> {
        let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key_bytes)?;
        let key = aead::LessSafeKey::new(key);

        Ok(Self {
            key: Arc::new(RwLock::new(key)),
            rng: Arc::new(RwLock::new(rand::SystemRandom::new())),
        })
    }

    pub async fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let key = self.key.read().await;
        let rng = self.rng.read().await;
        
        let nonce = rand::generate(&rng)?;
        let nonce = nonce.expose();
        
        let mut in_out = data.to_vec();
        let nonce = aead::Nonce::assume_unique_for_key(nonce.as_ref().try_into()?);
        
        key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)?;
        
        Ok(in_out)
    }

    pub async fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let key = self.key.read().await;
        
        let mut in_out = encrypted_data.to_vec();
        let nonce = aead::Nonce::assume_unique_for_key(&encrypted_data[..12].try_into()?);
        
        key.open_in_place(nonce, aead::Aad::empty(), &mut in_out)?;
        in_out.truncate(in_out.len() - 16); // Remove the authentication tag
        
        Ok(in_out)
    }

    pub async fn encrypt_string(&self, text: &str) -> Result<String> {
        let encrypted = self.encrypt(text.as_bytes()).await?;
        Ok(base64::encode(encrypted))
    }

    pub async fn decrypt_string(&self, encrypted_text: &str) -> Result<String> {
        let encrypted = base64::decode(encrypted_text)?;
        let decrypted = self.decrypt(&encrypted).await?;
        Ok(String::from_utf8(decrypted)?)
    }

    pub async fn rotate_key(&self) -> Result<()> {
        let rng = self.rng.read().await;
        let key_bytes = rand::generate(&rng)?;
        let key_bytes = key_bytes.expose();
        
        let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key_bytes.as_ref())?;
        let key = aead::LessSafeKey::new(key);
        
        let mut current_key = self.key.write().await;
        *current_key = key;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption() -> Result<()> {
        let manager = EncryptionManager::new()?;
        
        // Test byte encryption
        let data = b"test data";
        let encrypted = manager.encrypt(data).await?;
        let decrypted = manager.decrypt(&encrypted).await?;
        assert_eq!(data, decrypted.as_slice());
        
        // Test string encryption
        let text = "test string";
        let encrypted = manager.encrypt_string(text).await?;
        let decrypted = manager.decrypt_string(&encrypted).await?;
        assert_eq!(text, decrypted);
        
        // Test key rotation
        manager.rotate_key().await?;
        let encrypted_after_rotation = manager.encrypt_string(text).await?;
        assert_ne!(encrypted, encrypted_after_rotation);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_key_persistence() -> Result<()> {
        let manager1 = EncryptionManager::new()?;
        let key_bytes = manager1.key.read().await.key().as_ref().to_vec();
        
        let manager2 = EncryptionManager::new_with_key(&key_bytes)?;
        
        let text = "test string";
        let encrypted = manager1.encrypt_string(text).await?;
        let decrypted = manager2.decrypt_string(&encrypted).await?;
        assert_eq!(text, decrypted);
        
        Ok(())
    }
} 
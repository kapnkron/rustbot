use crate::utils::error::Result;
use ring::{aead, rand};
use std::path::Path;
use log::{info, warn};

pub struct SecureStorage {
    key: aead::LessSafeKey,
}

impl SecureStorage {
    pub fn new(key_path: &Path) -> Result<Self> {
        let rng = rand::SystemRandom::new();
        let key_bytes = ring::rand::generate(&rng)?;
        let key = aead::LessSafeKey::new(aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key_bytes.as_ref())?);
        Ok(Self { key })
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let nonce = rand::generate(&rand::SystemRandom::new())?;
        let mut in_out = data.to_vec();
        self.key.seal_in_place_append_tag(aead::Nonce::assume_unique_for_key(nonce.as_ref()), aead::Aad::empty(), &mut in_out)?;
        Ok(in_out)
    }

    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let nonce = rand::generate(&rand::SystemRandom::new())?;
        let mut in_out = data.to_vec();
        self.key.open_in_place(aead::Nonce::assume_unique_for_key(nonce.as_ref()), aead::Aad::empty(), &mut in_out)?;
        Ok(in_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_secure_storage() -> Result<()> {
        let storage = SecureStorage::new(&PathBuf::from("test_key"))?;
        let data = b"test data";
        let encrypted = storage.encrypt(data)?;
        let decrypted = storage.decrypt(&encrypted)?;
        assert_eq!(data, &decrypted[..]);
        Ok(())
    }
} 
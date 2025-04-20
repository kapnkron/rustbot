use crate::error::{Result, Error};
use ring::{aead, rand};
use ring::rand::SecureRandom;
use std::path::Path;
use log::{info, warn};
use std::array::TryFromSliceError;

impl From<TryFromSliceError> for Error {
    fn from(err: TryFromSliceError) -> Self {
        Error::SecurityError(format!("Slice conversion error: {}", err))
    }
}

pub struct SecureStorage {
    key: aead::LessSafeKey,
    rng: rand::SystemRandom,
}

impl SecureStorage {
    pub fn new(_key_path: &Path) -> Result<Self> {
        let rng = rand::SystemRandom::new();
        let mut key_bytes = vec![0u8; 32];
        rng.fill(&mut key_bytes)?;
        let key = aead::LessSafeKey::new(aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes)?);
        Ok(Self { key, rng })
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut nonce = vec![0u8; 12];
        self.rng.fill(&mut nonce)?;
        let mut in_out = data.to_vec();
        let nonce = aead::Nonce::assume_unique_for_key(nonce[..12].try_into()?);
        self.key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)?;
        Ok(in_out)
    }

    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 12 {
            return Err(Error::SecurityError("Data too short for decryption".to_string()));
        }
        let mut in_out = data.to_vec();
        let nonce = aead::Nonce::assume_unique_for_key(data[..12].try_into()?);
        self.key.open_in_place(nonce, aead::Aad::empty(), &mut in_out)?;
        in_out.truncate(in_out.len() - 16); // Remove the authentication tag
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
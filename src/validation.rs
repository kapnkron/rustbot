use crate::error::{Result, Error};

pub fn validate_api_key(api_key: &str) -> Result<()> {
    if api_key.is_empty() {
        return Err(Error::ConfigError("API key cannot be empty".to_string()));
    }
    if api_key.len() < 32 {
        return Err(Error::ConfigError("API key is too short".to_string()));
    }
    Ok(())
}

pub fn validate_symbol(symbol: &str) -> Result<()> {
    if symbol.is_empty() {
        return Err(Error::ValidationError("Symbol cannot be empty".to_string()));
    }
    if !symbol.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return Err(Error::ValidationError("Symbol must contain only uppercase letters and digits".to_string()));
    }
    Ok(())
}

pub fn validate_amount(amount: f64) -> Result<()> {
    if amount <= 0.0 {
        return Err(Error::ValidationError("Amount must be positive".to_string()));
    }
    if amount > 1_000_000.0 {
        return Err(Error::ValidationError("Amount is too large".to_string()));
    }
    Ok(())
} 
use crate::error::{Result, Error};
use regex::Regex;
use log::{info, warn};

pub struct InputValidator {
    max_length: usize,
    pattern: Regex,
}

impl InputValidator {
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            pattern: Regex::new(r"^[a-zA-Z0-9_\-\.@\s]+$").unwrap(),
        }
    }

    pub fn validate(&self, input: &str) -> Result<bool> {
        if input.len() > self.max_length {
            warn!("Input length {} exceeds maximum allowed length {}", input.len(), self.max_length);
            return Err(Error::ValidationError("Input too long".to_string()));
        }

        if !self.pattern.is_match(input) {
            warn!("Input contains invalid characters: {}", input);
            return Err(Error::ValidationError("Input contains invalid characters".to_string()));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_validator() -> Result<()> {
        let validator = InputValidator::new(100);
        assert!(validator.validate("valid_input")?);
        assert!(!validator.validate("invalid@input")?);
        assert!(!validator.validate(&"a".repeat(101))?);
        Ok(())
    }
} 
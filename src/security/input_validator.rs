use crate::error::Result;
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
            pattern: Regex::new(r"^[a-zA-Z0-9_\-\.]+$").unwrap(),
        }
    }

    pub fn validate(&self, input: &str) -> Result<bool> {
        Ok(input.len() <= self.max_length && self.pattern.is_match(input))
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
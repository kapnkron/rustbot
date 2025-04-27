use std::result::Result as StdResult;
use thiserror::Error;
use reqwest;
use teloxide::RequestError;
use serde_json;
use anyhow;
use prometheus;
use ring;
use std::io;
use tch::TchError;
use serde_urlencoded::de::Error as SerdeUrlencodedError;
use crate::ml::config::MLConfigError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("API invalid data: {0}")]
    ApiInvalidData(String),
    #[error("API invalid format: {0}")]
    ApiInvalidFormat(String),
    #[error("API connection failed: {0}")]
    ApiConnectionFailed(String),
    #[error("API authentication failed: {0}")]
    ApiAuthFailed(String),
    #[error("API quota exceeded: {0}")]
    ApiQuotaExceeded(String),
    #[error("API maintenance: {0}")]
    ApiMaintenance(String),
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Security error: {0}")]
    SecurityError(String),
    #[error("ML error: {0}")]
    MLError(String),
    #[error("ML config error: {0}")]
    MLConfigError(#[from] MLConfigError),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("Trading error: {0}")]
    TradingError(String),
    #[error("Solana RPC error: {0}")]
    SolanaRpcError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Keychain error: {0}")]
    KeychainError(String),
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::ApiInvalidFormat(err.to_string())
    }
}

impl From<RequestError> for Error {
    fn from(err: RequestError) -> Self {
        Error::ApiError(err.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::InternalError(err.to_string())
    }
}

impl From<prometheus::Error> for Error {
    fn from(err: prometheus::Error) -> Self {
        Error::InternalError(err.to_string())
    }
}

impl From<ring::error::Unspecified> for Error {
    fn from(err: ring::error::Unspecified) -> Self {
        Error::SecurityError(format!("{:?}", err))
    }
}

impl From<TchError> for Error {
    fn from(err: TchError) -> Self {
        Error::MLError(err.to_string())
    }
}

impl From<SerdeUrlencodedError> for Error {
    fn from(err: SerdeUrlencodedError) -> Self {
        Error::ParseError(err.to_string())
    }
}

impl From<crate::api::ApiError> for Error {
    fn from(err: crate::api::ApiError) -> Self {
        match err {
            crate::api::ApiError::RequestError(msg) => Error::ApiError(msg),
            crate::api::ApiError::RateLimitExceeded(msg) => Error::RateLimitExceeded(msg),
            crate::api::ApiError::ValidationError(msg) => Error::ValidationError(msg),
            crate::api::ApiError::InvalidFormat(msg) => Error::ApiInvalidFormat(msg),
        }
    }
}

pub type Result<T> = StdResult<T, Error>; 
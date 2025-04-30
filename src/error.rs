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
use solana_client::client_error::ClientError;
use solana_sdk::signer::keypair::SignerError;
use solana_program::program_error::ProgramError;
use keyring::Error as KeyringError;

#[derive(Error, Debug)]
pub enum Error {
    #[error("API Error: {0}")]
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
    #[error("Configuration Error: {0}")]
    ConfigError(String),
    #[error("Security Error: {0}")]
    SecurityError(String),
    #[error("ML error: {0}")]
    MLError(String),
    #[error("ML config error: {0}")]
    MLConfigError(#[from] MLConfigError),
    #[error("Internal Error: {0}")]
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
    #[error("HTTP Request Error: {0}")]
    HttpRequestError(#[from] reqwest::Error),
    #[error("Signing error: {0}")]
    SigningError(String),
    #[error("ML warmup required: {0}")]
    MLWarmupRequired(String),
    #[error("Data error: {0}")]
    DataError(String),
    #[error("Solana program error: {0}")]
    SolanaProgramError(#[from] ProgramError),
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::DataError(err.to_string())
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

impl From<ClientError> for Error {
    fn from(err: ClientError) -> Self {
        Error::SolanaRpcError(err.to_string())
    }
}

impl From<SignerError> for Error {
    fn from(err: SignerError) -> Self {
        Error::SigningError(err.to_string())
    }
}

impl From<KeyringError> for Error {
    fn from(err: KeyringError) -> Self {
        Error::KeychainError(err.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::InternalError(format!("Bincode Error: {}", err))
    }
}

impl From<base64::DecodeError> for Error {
    fn from(err: base64::DecodeError) -> Self {
        Error::InternalError(format!("Base64 Decode Error: {}", err))
    }
}

impl From<std::array::TryFromSliceError> for Error {
    fn from(err: std::array::TryFromSliceError) -> Self {
        Error::InternalError(format!("Slice conversion error: {}", err))
    }
}

pub type Result<T> = StdResult<T, Error>; 
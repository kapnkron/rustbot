use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

pub mod market;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: i64,
    pub symbol: String,
    pub side: String,
    pub amount: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
} 
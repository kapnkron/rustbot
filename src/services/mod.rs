use crate::error::Result;
use crate::models::{User, Trade};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct UserService {
    users: Arc<Mutex<Vec<User>>>,
}

impl UserService {
    pub fn new() -> Self {
        Self {
            users: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

pub struct TradeService {
    trades: Arc<Mutex<Vec<Trade>>>,
}

impl TradeService {
    pub fn new() -> Self {
        Self {
            trades: Arc::new(Mutex::new(Vec::new())),
        }
    }
} 
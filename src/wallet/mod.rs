use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub asset: String,
    pub free: f64,
    pub locked: f64,
    pub total: f64,
}

#[derive(Debug, Clone)]
pub struct Wallet {
    balances: Arc<Mutex<HashMap<String, Balance>>>,
}

impl Default for Wallet {
    fn default() -> Self {
        Self::new()
    }
}

impl Wallet {
    pub fn new() -> Self {
        Self {
            balances: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn get_balance(&self, asset: &str) -> Option<Balance> {
        self.balances.lock().await.get(asset).cloned()
    }

    pub async fn update_balance(&self, balance: Balance) {
        self.balances.lock().await.insert(balance.asset.clone(), balance);
    }

    pub async fn get_all_balances(&self) -> Vec<Balance> {
        self.balances.lock().await.values().cloned().collect()
    }
} 
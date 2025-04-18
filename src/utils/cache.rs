use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Cache<T> {
    data: Arc<Mutex<HashMap<String, (T, Instant)>>>,
    ttl: Duration,
}

impl<T: Clone> Cache<T> {
    pub fn new(ttl: i64) -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            ttl: Duration::from_secs(ttl.try_into().unwrap()),
        }
    }

    pub async fn get(&self, key: &str) -> Option<T> {
        let data = self.data.lock().await;
        if let Some((value, timestamp)) = data.get(key) {
            if timestamp.elapsed() < self.ttl {
                return Some(value.clone());
            }
        }
        None
    }

    pub async fn set(&self, key: String, value: T) {
        let mut data = self.data.lock().await;
        data.insert(key, (value, Instant::now()));
    }
} 
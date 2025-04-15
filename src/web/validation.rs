use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct RateLimiter {
    limits: HashMap<String, (Duration, Instant)>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
        }
    }

    pub async fn check(&mut self, key: &str, limit: Duration) -> bool {
        if let Some((duration, last_call)) = self.limits.get(key) {
            if last_call.elapsed() < *duration {
                return false;
            }
        }
        
        self.limits.insert(key.to_string(), (limit, Instant::now()));
        true
    }

    pub async fn wait_until_ready(&mut self, key: &str, limit: Duration) {
        while !self.check(key, limit).await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
} 
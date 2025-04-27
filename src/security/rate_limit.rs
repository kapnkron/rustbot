use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use log::warn;

#[derive(Debug)]
pub struct RateLimiter {
    requests: Mutex<HashMap<String, Vec<Instant>>>,
    max_requests: u32,
    window: Duration,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            requests: Mutex::new(HashMap::new()),
            max_requests,
            window,
        }
    }

    pub async fn check(&self, ip: &str) -> Result<bool> {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        
        // Clean up old requests
        self.cleanup_old_requests(&mut requests, now);
        
        // Get or create request history for this IP
        let ip_requests = requests.entry(ip.to_string())
            .or_insert_with(Vec::new);
        
        // Check if we're under the limit
        if ip_requests.len() >= self.max_requests as usize {
            warn!("Rate limit exceeded for IP: {}", ip);
            return Ok(false);
        }
        
        // Add new request
        ip_requests.push(now);
        Ok(true)
    }

    fn cleanup_old_requests(&self, requests: &mut HashMap<String, Vec<Instant>>, now: Instant) {
        let window_start = now - self.window;
        for timestamps in requests.values_mut() {
            timestamps.retain(|&time| time >= window_start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[tokio::test]
    async fn test_rate_limit() -> Result<()> {
        use std::net::IpAddr;
        use std::time::Duration;
        
        let limiter = RateLimiter::new(2, Duration::from_millis(100));
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        let ip_str = ip.to_string();

        assert!(limiter.check(&ip_str).await?);
        assert!(limiter.check(&ip_str).await?);
        assert!(!limiter.check(&ip_str).await?);

        // Wait for the limit to reset
        Ok(())
    }
} 
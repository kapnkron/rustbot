use crate::error::{Result, Error};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use log::{info, warn};
use std::net::IpAddr;

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
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_rate_limiter() -> Result<()> {
        let limiter = RateLimiter::new(2, Duration::from_secs(1));
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        
        assert!(limiter.check(ip).await?);
        assert!(limiter.check(ip).await?);
        assert!(!limiter.check(ip).await?);
        
        Ok(())
    }
} 
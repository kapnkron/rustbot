use crate::utils::error::Result;
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use log::{info, warn};

pub struct RateLimiter {
    requests: Mutex<HashMap<IpAddr, Vec<Instant>>>,
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

    pub async fn check_limit(&self, ip: IpAddr) -> Result<bool> {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        
        // Clean up old requests
        if let Some(timestamps) = requests.get_mut(&ip) {
            timestamps.retain(|&t| now.duration_since(t) < self.window);
            
            if timestamps.len() >= self.max_requests as usize {
                return Ok(false);
            }
            
            timestamps.push(now);
        } else {
            requests.insert(ip, vec![now]);
        }
        
        Ok(true)
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
        
        assert!(limiter.check_limit(ip).await?);
        assert!(limiter.check_limit(ip).await?);
        assert!(!limiter.check_limit(ip).await?);
        
        Ok(())
    }
} 
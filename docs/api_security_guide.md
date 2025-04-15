# API Security Guide

This guide provides detailed instructions for implementing and using the API security features in the trading system.

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [API Key Management](#api-key-management)
3. [Access Control](#access-control)
4. [Rate Limiting](#rate-limiting)
5. [Audit Logging](#audit-logging)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Prerequisites
- Rust 1.70 or later
- PostgreSQL database
- Redis (for rate limiting)

### Configuration
1. Add the following dependencies to your `Cargo.toml`:
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio"] }
redis = { version = "0.23", features = ["tokio-comp"] }
chrono = "0.4"
rand = "0.8"
```

2. Set up environment variables in `.env`:
```env
DATABASE_URL=postgres://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379
API_KEY_SECRET=your-secret-key-here
```

## API Key Management

### Generating API Keys
```rust
use trading_system::security::api::ApiSecurityManager;
use std::time::Duration;

async fn generate_api_key() -> Result<(), Box<dyn std::error::Error>> {
    let api_security = ApiSecurityManager::new(
        auth_manager,
        authz_manager,
        rate_limiter,
        audit_logger,
    );

    let api_key = api_security.generate_api_key(
        "Trading API".to_string(),
        "API key for trading operations".to_string(),
        vec!["read".to_string(), "trade".to_string()],
        Some((100, Duration::minutes(1))),
        Some(Duration::days(30)),
    ).await?;

    println!("Generated API key: {}", api_key.key);
    Ok(())
}
```

### Validating API Keys
```rust
async fn validate_api_key(api_key: &str, permission: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let api_security = ApiSecurityManager::new(
        auth_manager,
        authz_manager,
        rate_limiter,
        audit_logger,
    );

    let is_valid = api_security.validate_api_key(api_key, permission).await?;
    Ok(is_valid)
}
```

### Revoking API Keys
```rust
async fn revoke_api_key(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let api_security = ApiSecurityManager::new(
        auth_manager,
        authz_manager,
        rate_limiter,
        audit_logger,
    );

    api_security.revoke_api_key(api_key).await?;
    Ok(())
}
```

## Access Control

### Permission Management
1. Define available permissions in your application:
```rust
const PERMISSIONS: &[&str] = &[
    "read",
    "trade",
    "admin",
    "report",
];
```

2. Check permissions in your API endpoints:
```rust
async fn handle_trade_request(api_key: &str, trade_request: TradeRequest) -> Result<(), Box<dyn std::error::Error>> {
    let api_security = ApiSecurityManager::new(
        auth_manager,
        authz_manager,
        rate_limiter,
        audit_logger,
    );

    if !api_security.validate_api_key(api_key, "trade").await? {
        return Err("Insufficient permissions".into());
    }

    // Process trade request
    Ok(())
}
```

## Rate Limiting

### Configuring Rate Limits
1. Set default rate limits:
```rust
let rate_limiter = RateLimiter::new(
    Duration::minutes(1),
    100, // requests per minute
);
```

2. Set custom rate limits for specific API keys:
```rust
let api_key = api_security.generate_api_key(
    "High Frequency Trading".to_string(),
    "API key for high frequency trading".to_string(),
    vec!["trade".to_string()],
    Some((1000, Duration::minutes(1))), // 1000 requests per minute
    None,
).await?;
```

## Audit Logging

### Viewing Audit Logs
```rust
async fn view_audit_logs() -> Result<(), Box<dyn std::error::Error>> {
    let audit_logger = AuditLogger::new();
    
    let logs = audit_logger.get_events(
        Some(EventType::ApiKeyGenerated),
        Some(Severity::Info),
        Some(Utc::now() - Duration::days(7)),
        Some(Utc::now()),
    ).await?;

    for log in logs {
        println!("{:?}", log);
    }
    
    Ok(())
}
```

## Best Practices

1. **API Key Security**
   - Never log API keys
   - Use HTTPS for all API calls
   - Implement key rotation
   - Set appropriate expiration times
   - Monitor for suspicious activity

2. **Rate Limiting**
   - Set appropriate limits based on use case
   - Monitor rate limit usage
   - Implement gradual limit increases
   - Log rate limit violations
   - Consider burst limits

3. **Access Control**
   - Use principle of least privilege
   - Regularly review permissions
   - Implement role-based access
   - Log permission changes
   - Monitor access patterns

4. **Audit Logging**
   - Log all security events
   - Implement log rotation
   - Secure log storage
   - Regular log analysis
   - Alert on suspicious patterns

## Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Check key format
   - Verify key is not expired
   - Confirm key is active
   - Check key permissions
   - Verify key signature

2. **Rate Limit Exceeded**
   - Check current rate limit
   - Verify request frequency
   - Consider increasing limit
   - Implement backoff strategy
   - Monitor usage patterns

3. **Permission Denied**
   - Verify required permissions
   - Check user role
   - Review permission assignments
   - Check permission hierarchy
   - Verify permission scope

4. **Audit Log Issues**
   - Check log storage
   - Verify log permissions
   - Check log rotation
   - Monitor log size
   - Verify log format

### Debugging Tips
1. Enable debug logging:
```rust
env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
```

2. Check API key status:
```rust
let keys = api_security.list_api_keys().await?;
for key in keys {
    println!("Key: {}, Active: {}, Expires: {:?}", 
        key.name, 
        key.is_active, 
        key.expires_at
    );
}
```

3. Monitor rate limits:
```rust
let usage = rate_limiter.get_usage(api_key).await?;
println!("Current usage: {}/{}", usage.current, usage.limit);
```

## Support

For additional support:
1. Check the documentation
2. Review error logs
3. Contact security team
4. Submit bug reports
5. Request feature enhancements 
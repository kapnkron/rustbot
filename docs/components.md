# Components Documentation

## API Module

### MarketDataCollector
The `MarketDataCollector` is a high-level component that aggregates market data from multiple sources (CoinGecko, CoinMarketCap, and CryptoDataDownload). It provides weighted averages of market data to ensure reliable and accurate information.

#### Usage
```rust
let collector = MarketDataCollector::new(
    "coingecko_api_key".to_string(),
    "coinmarketcap_api_key".to_string(),
    "cryptodatadownload_api_key".to_string(),
);

let market_data = collector.collect_market_data("BTC").await?;
```

### CoinGeckoClient
The `CoinGeckoClient` provides access to the CoinGecko API for cryptocurrency market data.

#### Usage
```rust
let rate_limiter = Arc::new(Mutex::new(RateLimiter::new()));
let client = CoinGeckoClient::new("api_key".to_string(), rate_limiter);

// Get market data
let market_data = client.get_market_data("BTC").await?;

// Get token info
let token_info = client.get_token_info("bitcoin").await?;

// Get price history
let price_history = client.get_price_history("bitcoin", 30).await?;
```

### RateLimiter
The `RateLimiter` is a utility component that helps manage API rate limits by tracking and enforcing request intervals.

#### Usage
```rust
let mut rate_limiter = RateLimiter::new();

// Check if a request can be made
if rate_limiter.check("api_key", Duration::from_secs(1)).await {
    // Make request
}

// Wait until a request can be made
rate_limiter.wait_until_ready("api_key", Duration::from_secs(1)).await;
```

## Error Handling

The error handling system uses a custom `Error` enum with various error types for different scenarios:

- `ApiError`: General API errors
- `ApiInvalidData`: Invalid data received from API
- `ApiInvalidFormat`: Invalid response format
- `ApiConnectionFailed`: Network connection issues
- `ApiAuthFailed`: Authentication failures
- `ApiQuotaExceeded`: API quota exceeded
- `ApiMaintenance`: API under maintenance
- `RateLimitExceeded`: Rate limit exceeded
- `NetworkError`: General network errors
- `ParseError`: Data parsing errors
- `ValidationError`: Data validation errors

#### Usage
```rust
use crate::utils::error::{Result, Error};

fn example() -> Result<()> {
    // Return specific error types
    Err(Error::ApiError("Something went wrong".to_string()))
}
```

## Data Types

### MarketData
Represents market data for a cryptocurrency:
```rust
pub struct MarketData {
    pub price: f64,
    pub volume: f64,
    pub market_cap: f64,
    pub price_change_24h: f64,
    pub volume_change_24h: f64,
}
```

### Quote
Represents a cryptocurrency quote:
```rust
pub struct Quote {
    #[serde(rename = "USD")]
    pub usd: USDData,
}
```

### USDData
Represents USD-specific data for a cryptocurrency:
```rust
pub struct USDData {
    pub price: f64,
    pub volume_24h: f64,
    pub market_cap: f64,
    pub percent_change_24h: f64,
    pub volume_change_24h: f64,
}
``` 
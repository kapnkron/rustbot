use prometheus::{Counter, Gauge, Histogram, Registry};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    pub static ref API_CALLS: Counter = Counter::new(
        "api_calls_total",
        "Total number of API calls"
    ).unwrap();
    
    pub static ref API_ERRORS: Counter = Counter::new(
        "api_errors_total",
        "Total number of API errors"
    ).unwrap();
    
    pub static ref MARKET_PRICE: Gauge = Gauge::new(
        "market_price",
        "Current market price"
    ).unwrap();
    
    pub static ref API_LATENCY: Histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new(
            "api_latency_seconds",
            "API call latency in seconds"
        ).buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0])
    ).unwrap();
}

pub fn init() -> Result<(), prometheus::Error> {
    REGISTRY.register(Box::new(API_CALLS.clone()))?;
    REGISTRY.register(Box::new(API_ERRORS.clone()))?;
    REGISTRY.register(Box::new(MARKET_PRICE.clone()))?;
    REGISTRY.register(Box::new(API_LATENCY.clone()))?;
    Ok(())
} 
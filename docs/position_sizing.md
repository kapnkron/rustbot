# Position Sizing Documentation

## Overview
This document describes the position sizing logic implemented in the trading system, which is a critical component of risk management.

## Position Sizing Strategy

### 1. Risk-Based Position Sizing
The system uses a risk-based approach to determine position sizes, considering:
- Portfolio value
- Risk per trade (as percentage of portfolio)
- Stop loss distance
- Maximum position size limit

### 2. Key Components

#### Position Size Calculation
```rust
pub async fn calculate_position_size(
    &self,
    symbol: &str,
    entry_price: f64,
    stop_loss: f64,
) -> Result<f64, PortfolioError> {
    let portfolio_value = self.get_portfolio_value().await?;
    let risk_amount = portfolio_value * self.config.risk_per_trade;
    let position_size_limit = portfolio_value * self.config.position_size_limit;

    // Calculate position size based on risk
    let price_risk = (entry_price - stop_loss).abs();
    let position_size = risk_amount / price_risk;

    // Ensure position size doesn't exceed limits
    let position_value = position_size * entry_price;
    if position_value > position_size_limit {
        return Ok(position_size_limit / entry_price);
    }

    Ok(position_size)
}
```

### 3. Configuration Parameters

```rust
pub struct PortfolioConfig {
    pub initial_balance: f64,
    pub base_currency: String,
    pub max_positions: usize,
    pub position_size_limit: f64, // Maximum position size as percentage of portfolio
    pub risk_per_trade: f64,      // Risk per trade as percentage of portfolio
    pub stop_loss_percentage: f64,
}
```

## Usage Examples

### Basic Position Sizing
```rust
let config = PortfolioConfig {
    initial_balance: 100000.0,
    base_currency: "USD".to_string(),
    max_positions: 10,
    position_size_limit: 0.2, // 20% of portfolio
    risk_per_trade: 0.01,     // 1% risk per trade
    stop_loss_percentage: 0.05,
};

let portfolio = PortfolioManager::new(config);

// Calculate position size for BTC/USD
let position_size = portfolio
    .calculate_position_size(
        "BTC/USD",
        50000.0,  // Entry price
        47500.0,  // Stop loss
    )
    .await?;
```

### Position Management
```rust
// Open a position
portfolio.open_position(
    "BTC/USD".to_string(),
    position_size,
    50000.0,
    PositionSide::Long,
).await?;

// Close a position
let pnl = portfolio.close_position("BTC/USD", 51000.0).await?;
```

## Best Practices

1. **Risk Management**
   - Never risk more than 1-2% of portfolio per trade
   - Use appropriate stop losses
   - Consider correlation between positions
   - Monitor portfolio exposure

2. **Position Sizing**
   - Calculate position size before entry
   - Consider market volatility
   - Account for slippage
   - Monitor position limits

3. **Portfolio Management**
   - Regular portfolio rebalancing
   - Monitor position correlations
   - Track performance metrics
   - Maintain position records

## Error Handling
The position sizing system uses the `Result` type for error handling. Common errors include:
- Invalid position size
- Insufficient funds
- Position limit exceeded
- Invalid asset allocation
- Calculation errors

## Testing
The system includes comprehensive tests for:
- Position size calculation
- Risk management
- Position limits
- Error handling
- Performance metrics

Run tests with:
```bash
cargo test --package trading_system --lib trading::portfolio
```

## Next Steps
1. Implement advanced position sizing strategies
2. Add correlation-based position sizing
3. Implement dynamic position sizing
4. Add position sizing optimization
5. Implement position sizing backtesting 
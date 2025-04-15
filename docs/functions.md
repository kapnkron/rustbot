# Trading Bot Function Documentation

## UI Configuration

### UIConfig
Central configuration structure containing all user-configurable parameters.

#### RiskUIConfig
- `max_drawdown`: Maximum allowed drawdown (0.0 to 1.0)
- `max_position_size`: Maximum size for a single position
- `max_total_exposure`: Maximum total exposure across all positions
- `max_leverage`: Maximum allowed leverage
- `risk_per_trade`: Risk percentage per trade
- `max_volatility`: Maximum allowed volatility
- `min_liquidity`: Minimum required liquidity
- `max_daily_trades`: Maximum number of trades per day
- `max_open_positions`: Maximum number of open positions

#### OrderUIConfig
- `default_pair`: Default trading pair
- `default_size`: Default order size
- `default_leverage`: Default leverage
- `default_expiry_hours`: Default order expiry time
- `max_slippage`: Maximum allowed slippage
- `max_spread`: Maximum allowed spread

#### SignalUIConfig
- `min_strength`: Minimum signal strength
- `max_strength`: Maximum signal strength
- `required_indicators`: List of required technical indicators
- `risk_reward_ratio`: Required risk/reward ratio
- `min_volume`: Minimum trading volume
- `max_spread`: Maximum allowed spread

#### ExchangeUIConfig
- `default_exchange`: Default exchange name
- `api_key`: Exchange API key
- `api_secret`: Exchange API secret
- `testnet`: Whether to use testnet
- `pairs`: List of trading pairs

#### MLUIConfig
- `model_path`: Path to ML model file
- `training_interval`: Model retraining interval in seconds
- `feature_window`: Number of historical data points for features
- `prediction_horizon`: Number of future points to predict
- `confidence_threshold`: Minimum confidence for predictions
- `retrain_threshold`: Threshold for model retraining
- `feature_weights`: Weights for different features
- `enabled_features`: List of enabled features
- `use_risk_features`: Whether to use risk features
- `use_order_features`: Whether to use order features
- `use_signal_features`: Whether to use signal features
- `use_market_features`: Whether to use market features

## UI Manager

### UIManager
Manages UI configuration and provides access to configuration parameters.

#### Configuration Management
- `new()`: Creates a new UI manager with default configuration
- `update_risk_config()`: Updates risk configuration
- `update_order_config()`: Updates order configuration
- `update_signal_config()`: Updates signal configuration
- `update_exchange_config()`: Updates exchange configuration
- `update_ml_config()`: Updates ML configuration
- `get_risk_config()`: Retrieves risk configuration
- `get_order_config()`: Retrieves order configuration
- `get_signal_config()`: Retrieves signal configuration
- `get_exchange_config()`: Retrieves exchange configuration
- `get_ml_config()`: Retrieves ML configuration
- `validate_config()`: Validates all configuration parameters

#### Feature Access
- `get_risk_features()`: Retrieves risk-related features for ML
- `get_order_features()`: Retrieves order-related features for ML
- `get_signal_features()`: Retrieves signal-related features for ML

## Error Handling

### TradingError
Enum containing all possible trading-related errors.

#### Variants
- `InvalidConfig`: Invalid configuration parameter
- `RateLimitExceeded`: API rate limit exceeded
- `RiskLimitExceeded`: Risk limit exceeded
- `InsufficientData`: Insufficient data for operation
- `DeviceNotFound`: Hardware device not found
- `HealthError`: Health check failed
- `WalletError`: Wallet operation failed

## Current Issues to Fix

1. Error Type Mismatches:
   - `RateLimitExceeded` needs String argument
   - `BonkError` needs to be wrapped in `TradingError`
   - `MetricsError` needs to be wrapped in `TradingError`
   - `YubiKeyError` needs to be wrapped in `TradingError`

2. Missing Fields:
   - `rsi_period` in `BonkConfig`
   - `data_provider` in `TradingStrategy`
   - `pair_max_leverages` in `RiskConfig`
   - `exit_price` in `Position`

3. Async/Await Issues:
   - Missing `.await` on `check_risk_limits`
   - Missing `.await` on `calculate_position_size`

4. Borrowing Issues:
   - Borrow of moved value `name` in `health.rs`
   - Borrow of moved value `strategy` in `strategy.rs`

5. Missing Dependencies:
   - `tch` crate for ML operations
   - `Axis` function for ML operations

## Next Steps

1. Fix error type mismatches
2. Add missing fields to structs
3. Fix async/await issues
4. Resolve borrowing issues
5. Add missing dependencies
6. Implement ML model integration
7. Add comprehensive testing
8. Create user documentation 
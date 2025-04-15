use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum PortfolioError {
    #[error("Invalid position size")]
    InvalidPositionSize,
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("Position limit exceeded")]
    PositionLimitExceeded,
    #[error("Invalid asset allocation")]
    InvalidAllocation,
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub timestamp: SystemTime,
    pub side: PositionSide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConfig {
    pub initial_balance: f64,
    pub base_currency: String,
    pub max_positions: usize,
    pub position_size_limit: f64, // Maximum position size as percentage of portfolio
    pub risk_per_trade: f64,      // Risk per trade as percentage of portfolio
    pub stop_loss_percentage: f64,
}

pub struct PortfolioManager {
    config: PortfolioConfig,
    positions: RwLock<HashMap<String, Position>>,
    balance: RwLock<f64>,
    performance_metrics: RwLock<PerformanceMetrics>,
}

#[derive(Debug, Default)]
struct PerformanceMetrics {
    total_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
}

impl PortfolioManager {
    pub fn new(config: PortfolioConfig) -> Self {
        Self {
            config,
            positions: RwLock::new(HashMap::new()),
            balance: RwLock::new(config.initial_balance),
            performance_metrics: RwLock::new(PerformanceMetrics::default()),
        }
    }

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

    pub async fn open_position(
        &self,
        symbol: String,
        quantity: f64,
        entry_price: f64,
        side: PositionSide,
    ) -> Result<(), PortfolioError> {
        let mut positions = self.positions.write().await;
        let mut balance = self.balance.write().await;

        // Check position limits
        if positions.len() >= self.config.max_positions {
            return Err(PortfolioError::PositionLimitExceeded);
        }

        // Calculate position value
        let position_value = quantity * entry_price;
        if position_value > *balance {
            return Err(PortfolioError::InsufficientFunds);
        }

        // Create and store position
        let position = Position {
            symbol: symbol.clone(),
            quantity,
            entry_price,
            current_price: entry_price,
            timestamp: SystemTime::now(),
            side,
        };

        positions.insert(symbol, position);
        *balance -= position_value;

        Ok(())
    }

    pub async fn close_position(
        &self,
        symbol: &str,
        exit_price: f64,
    ) -> Result<f64, PortfolioError> {
        let mut positions = self.positions.write().await;
        let mut balance = self.balance.write().await;

        if let Some(position) = positions.remove(symbol) {
            let pnl = match position.side {
                PositionSide::Long => (exit_price - position.entry_price) * position.quantity,
                PositionSide::Short => (position.entry_price - exit_price) * position.quantity,
            };

            *balance += pnl + (position.quantity * position.entry_price);
            self.update_performance_metrics(pnl).await;

            Ok(pnl)
        } else {
            Err(PortfolioError::InvalidPositionSize)
        }
    }

    pub async fn get_portfolio_value(&self) -> Result<f64, PortfolioError> {
        let positions = self.positions.read().await;
        let balance = self.balance.read().await;

        let positions_value: f64 = positions
            .values()
            .map(|p| p.quantity * p.current_price)
            .sum();

        Ok(*balance + positions_value)
    }

    async fn update_performance_metrics(&self, pnl: f64) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_return += pnl;
        // Update other metrics as needed
    }

    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    pub async fn rebalance_portfolio(
        &self,
        target_allocations: HashMap<String, f64>,
    ) -> Result<(), PortfolioError> {
        let current_value = self.get_portfolio_value().await?;
        let positions = self.positions.read().await;

        for (symbol, target_allocation) in target_allocations {
            let current_allocation = positions
                .get(&symbol)
                .map(|p| (p.quantity * p.current_price) / current_value)
                .unwrap_or(0.0);

            let allocation_diff = target_allocation - current_allocation;
            if allocation_diff.abs() > 0.01 { // 1% threshold
                // Implement rebalancing logic here
                // This would involve calculating the required trades
                // and executing them through the order execution system
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_position_sizing() {
        let config = PortfolioConfig {
            initial_balance: 100000.0,
            base_currency: "USD".to_string(),
            max_positions: 10,
            position_size_limit: 0.2, // 20% of portfolio
            risk_per_trade: 0.01,     // 1% risk per trade
            stop_loss_percentage: 0.05,
        };

        let portfolio = PortfolioManager::new(config);
        
        // Test position size calculation
        let position_size = portfolio
            .calculate_position_size("BTC/USD", 50000.0, 47500.0)
            .await
            .unwrap();
        
        assert!(position_size > 0.0);
        assert!(position_size * 50000.0 <= 20000.0); // Should not exceed 20% of portfolio
    }

    #[tokio::test]
    async fn test_position_management() {
        let config = PortfolioConfig {
            initial_balance: 100000.0,
            base_currency: "USD".to_string(),
            max_positions: 10,
            position_size_limit: 0.2,
            risk_per_trade: 0.01,
            stop_loss_percentage: 0.05,
        };

        let portfolio = PortfolioManager::new(config);
        
        // Test opening position
        portfolio
            .open_position(
                "BTC/USD".to_string(),
                0.5,
                50000.0,
                PositionSide::Long,
            )
            .await
            .unwrap();

        // Test closing position
        let pnl = portfolio
            .close_position("BTC/USD", 51000.0)
            .await
            .unwrap();
        
        assert_eq!(pnl, 500.0); // (51000 - 50000) * 0.5
    }
} 
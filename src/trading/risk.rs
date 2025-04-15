use crate::utils::error::Result;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiskManager {
    max_position_size: f64,
    max_risk_per_trade: f64,
    max_daily_loss: f64,
    position_sizes: HashMap<String, f64>,
    pub daily_pnl: f64,
}

impl RiskManager {
    pub fn new(max_position_size: f64, max_risk_per_trade: f64, max_daily_loss: f64) -> Self {
        Self {
            max_position_size,
            max_risk_per_trade,
            max_daily_loss,
            position_sizes: HashMap::new(),
            daily_pnl: 0.0,
        }
    }

    pub fn calculate_position_size(
        &self,
        current_price: f64,
        stop_loss: f64,
        account_balance: f64,
    ) -> f64 {
        let risk_amount = account_balance * self.max_risk_per_trade;
        let price_risk = (current_price - stop_loss).abs();
        let position_size = risk_amount / price_risk;
        
        // Ensure position size doesn't exceed maximum
        position_size.min(self.max_position_size)
    }

    pub fn update_daily_pnl(&mut self, pnl: f64) -> Result<()> {
        self.daily_pnl += pnl;
        if self.daily_pnl < -self.max_daily_loss {
            return Err(anyhow::anyhow!("Daily loss limit exceeded"));
        }
        Ok(())
    }

    pub fn can_open_position(&self, symbol: &str, proposed_size: f64) -> bool {
        let current_size = self.position_sizes.get(symbol).unwrap_or(&0.0);
        (current_size + proposed_size) <= self.max_position_size
    }

    pub fn update_position_size(&mut self, symbol: String, size: f64) {
        self.position_sizes.insert(symbol, size);
    }

    pub fn get_position_size(&self, symbol: &str) -> f64 {
        *self.position_sizes.get(symbol).unwrap_or(&0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_size_calculation() {
        let risk_manager = RiskManager::new(1000.0, 0.02, 0.05);
        let position_size = risk_manager.calculate_position_size(100.0, 98.0, 10000.0);
        
        // Risk amount = 10000 * 0.02 = 200
        // Price risk = 100 - 98 = 2
        // Position size = 200 / 2 = 100
        assert_eq!(position_size, 100.0);
    }

    #[test]
    fn test_position_size_limits() {
        let risk_manager = RiskManager::new(50.0, 0.02, 0.05);
        let position_size = risk_manager.calculate_position_size(100.0, 98.0, 10000.0);
        
        // Should be limited by max_position_size
        assert_eq!(position_size, 50.0);
    }

    #[test]
    fn test_daily_loss_limit() {
        let mut risk_manager = RiskManager::new(1000.0, 0.02, 0.05);
        
        // Initial loss within limit
        assert!(risk_manager.update_daily_pnl(-0.04).is_ok());
        
        // Additional loss that exceeds limit
        assert!(risk_manager.update_daily_pnl(-0.02).is_err());
    }

    #[test]
    fn test_position_size_tracking() {
        let mut risk_manager = RiskManager::new(1000.0, 0.02, 0.05);
        
        // Add position
        risk_manager.update_position_size("BTC".to_string(), 100.0);
        assert_eq!(risk_manager.get_position_size("BTC"), 100.0);
        
        // Update position
        risk_manager.update_position_size("BTC".to_string(), 200.0);
        assert_eq!(risk_manager.get_position_size("BTC"), 200.0);
        
        // Check non-existent position
        assert_eq!(risk_manager.get_position_size("ETH"), 0.0);
    }

    #[test]
    fn test_can_open_position() {
        let mut risk_manager = RiskManager::new(1000.0, 0.02, 0.05);
        
        // No existing position
        assert!(risk_manager.can_open_position("BTC", 500.0));
        
        // Add position
        risk_manager.update_position_size("BTC".to_string(), 500.0);
        
        // Can add more within limit
        assert!(risk_manager.can_open_position("BTC", 400.0));
        
        // Cannot exceed limit
        assert!(!risk_manager.can_open_position("BTC", 600.0));
    }
} 
use crate::error::Result;
use crate::error::Error;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiskManager {
    max_position_size: f64,
    // Comment out unused field
    // max_risk_per_trade: f64,
    max_daily_loss: f64,
    position_sizes: HashMap<String, f64>,
    pub daily_pnl: f64,
    // Fields likely added by mistake in previous edit, keep commented/remove if not needed
    // stop_loss_percentage: f64,
    // take_profit_percentage: f64,
    // daily_loss_limit_percentage: f64,
    // current_daily_pnl: f64,
}

impl RiskManager {
    pub fn new(max_position_size: f64, _max_risk_per_trade: f64, max_daily_loss: f64) -> Self {
        Self {
            max_position_size,
            // max_risk_per_trade, // Keep commented
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
        risk_factor: f64,
    ) -> f64 {
        if current_price <= stop_loss || current_price <= 0.0 || account_balance <= 0.0 {
            return 0.0;
        }

        let risk_amount_usd = account_balance * risk_factor;
        let price_risk_per_unit = current_price - stop_loss;
        
        let max_units_for_risk = risk_amount_usd / price_risk_per_unit;
        
        let desired_position_usd = max_units_for_risk * current_price;

        let final_position_usd = desired_position_usd.min(self.max_position_size);

        final_position_usd.max(0.0)
    }

    pub fn update_daily_pnl(&mut self, pnl_change: f64) -> Result<()> {
        self.daily_pnl += pnl_change;
        if self.daily_pnl < -self.max_daily_loss {
            // Log warning or error here? Or let caller handle Result?
            // For now, just return Error.
            Err(Error::ValidationError(format!(
                "Daily loss limit exceeded: current PnL {:.2}, limit {:.2}",
                self.daily_pnl, -self.max_daily_loss
            )))
        } else {
            // Add info log for successful update
            log::info!(
                "Updated daily PnL by {:.2}. New daily PnL: {:.2}", 
                pnl_change, self.daily_pnl
            );
            Ok(())
        }
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
        let risk_manager = RiskManager::new(1000.0, 0.02, 500.0);
        let balance = 10000.0;
        let price = 100.0;
        let stop_loss = 98.0;
        let risk_factor = 0.02;

        let expected_size_usd = 1000.0;
        let calculated_size_usd = risk_manager.calculate_position_size(
            price, stop_loss, balance, risk_factor
        );
        assert!((calculated_size_usd - expected_size_usd).abs() < 1e-6);
    }

    #[test]
    fn test_position_size_limits() {
        let risk_manager = RiskManager::new(500.0, 0.02, 500.0);
        let balance = 10000.0;
        let price = 100.0;
        let stop_loss = 98.0;
        let risk_factor = 0.02;

        let expected_size_usd = 500.0;
        let calculated_size_usd = risk_manager.calculate_position_size(
            price, stop_loss, balance, risk_factor
        );
        assert!((calculated_size_usd - expected_size_usd).abs() < 1e-6);
    }

    #[test]
    fn test_daily_loss_limit() {
        let mut risk_manager = RiskManager::new(1000.0, 0.02, 500.0); // Use absolute loss limit
        
        // Initial loss within limit
        assert!(risk_manager.update_daily_pnl(-400.0).is_ok());
        assert_eq!(risk_manager.daily_pnl, -400.0);
        
        // Additional loss that exceeds limit
        let result = risk_manager.update_daily_pnl(-200.0);
        assert!(result.is_err());
        assert_eq!(risk_manager.daily_pnl, -600.0); // PnL is still updated even if limit is hit

        // Adding profit brings it back within limits
        assert!(risk_manager.update_daily_pnl(150.0).is_ok());
        assert_eq!(risk_manager.daily_pnl, -450.0);
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
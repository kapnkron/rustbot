use crate::error::Result;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiskManager {
    max_position_size: f64,
    position_sizes: HashMap<String, f64>,
    pub daily_pnl: f64,
    // Fields likely added by mistake in previous edit, keep commented/remove if not needed
    // daily_loss_limit_percentage: f64,
    // current_daily_pnl: f64,
}

impl RiskManager {
    pub fn new(max_position_size: f64) -> Self {
        Self {
            max_position_size,
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
        // Simply log the update for now
        log::info!(
            "Updated daily PnL by {:.2}. New daily PnL: {:.2}", 
            pnl_change, self.daily_pnl
        );
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
        let risk_manager = RiskManager::new(1000.0);
        let balance = 10000.0;
        let price = 100.0;
        let stop_loss = 98.0;
        let risk_factor = 0.01;

        let expected_size_usd = 1000.0;
        let calculated_size_usd = risk_manager.calculate_position_size(
            price, stop_loss, balance, risk_factor
        );
        assert!((calculated_size_usd - expected_size_usd).abs() < 1e-6);
    }

    #[test]
    fn test_position_size_limits() {
        let risk_manager = RiskManager::new(500.0);
        let balance = 10000.0;
        let price = 100.0;
        let stop_loss = 98.0;
        let risk_factor = 0.01;

        let expected_size_usd = 500.0;
        let calculated_size_usd = risk_manager.calculate_position_size(
            price, stop_loss, balance, risk_factor
        );
        assert!((calculated_size_usd - expected_size_usd).abs() < 1e-6);
    }

    #[test]
    fn test_daily_loss_limit() {
        let mut risk_manager = RiskManager::new(1000.0);
        
        assert!(risk_manager.update_daily_pnl(-100.0).is_ok());
        assert_eq!(risk_manager.daily_pnl, -100.0);
        
        assert!(risk_manager.update_daily_pnl(50.0).is_ok());
        assert_eq!(risk_manager.daily_pnl, -50.0);
    }

    #[test]
    fn test_position_size_tracking() {
        let mut risk_manager = RiskManager::new(1000.0);
        
        risk_manager.update_position_size("BTC".to_string(), 100.0);
        assert_eq!(risk_manager.get_position_size("BTC"), 100.0);
        
        risk_manager.update_position_size("BTC".to_string(), 200.0);
        assert_eq!(risk_manager.get_position_size("BTC"), 200.0);
        
        assert_eq!(risk_manager.get_position_size("ETH"), 0.0);
    }

    #[test]
    fn test_can_open_position() {
        let mut risk_manager = RiskManager::new(1000.0);
        
        assert!(risk_manager.can_open_position("BTC", 500.0));
        
        risk_manager.update_position_size("BTC".to_string(), 500.0);
        
        assert!(risk_manager.can_open_position("BTC", 400.0));
        
        assert!(!risk_manager.can_open_position("BTC", 600.0));
    }
} 
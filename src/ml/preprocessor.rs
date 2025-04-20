pub fn process_market_data(&mut self, data: &TradingMarketData) -> Result<Vec<f64>> {
    let market_data: MarketData = data.into();
    // ... existing code ...
} 
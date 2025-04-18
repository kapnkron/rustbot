use crate::error::Result;
use crate::api::MarketDataCollector;
use teloxide::types::Message;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct TradingBot {
    market_data_collector: Arc<Mutex<MarketDataCollector>>,
}

impl TradingBot {
    pub fn new(
        coingecko_api_key: String,
        coinmarketcap_api_key: String,
        cryptodatadownload_api_key: String,
    ) -> Self {
        Self {
            market_data_collector: Arc::new(Mutex::new(MarketDataCollector::new(
                coingecko_api_key,
                coinmarketcap_api_key,
                cryptodatadownload_api_key,
            ))),
        }
    }

    pub async fn handle_message(&self, msg: Message) -> Result<()> {
        // TODO: Implement message handling logic
        Ok(())
    }
} 
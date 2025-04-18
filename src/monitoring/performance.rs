use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use log::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMetrics {
    pub total_trades: u64,
    pub successful_trades: u64,
    pub failed_trades: u64,
    pub total_profit: f64,
    pub total_loss: f64,
    pub average_trade_duration: Duration,
    pub win_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetrics {
    pub total_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: Duration,
    pub last_response_time: Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbMetrics {
    pub total_queries: u64,
    pub slow_queries: u64,
    pub average_query_time: Duration,
    pub last_query_time: Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlMetrics {
    pub total_predictions: u64,
    pub average_inference_time: Duration,
    pub last_inference_time: Duration,
    pub prediction_accuracy: f64,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub trade_metrics: TradeMetrics,
    pub api_metrics: ApiMetrics,
    pub db_metrics: DbMetrics,
    pub ml_metrics: MlMetrics,
}

pub struct PerformanceMonitor {
    metrics: RwLock<PerformanceMetrics>,
    trade_start_times: RwLock<HashMap<String, Instant>>,
    api_start_times: RwLock<HashMap<String, Instant>>,
    db_start_times: RwLock<HashMap<String, Instant>>,
    ml_start_times: RwLock<HashMap<String, Instant>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(PerformanceMetrics {
                timestamp: Utc::now(),
                trade_metrics: TradeMetrics {
                    total_trades: 0,
                    successful_trades: 0,
                    failed_trades: 0,
                    total_profit: 0.0,
                    total_loss: 0.0,
                    average_trade_duration: Duration::from_secs(0),
                    win_rate: 0.0,
                },
                api_metrics: ApiMetrics {
                    total_requests: 0,
                    failed_requests: 0,
                    average_response_time: Duration::from_secs(0),
                    last_response_time: Duration::from_secs(0),
                    error_rate: 0.0,
                },
                db_metrics: DbMetrics {
                    total_queries: 0,
                    slow_queries: 0,
                    average_query_time: Duration::from_secs(0),
                    last_query_time: Duration::from_secs(0),
                    error_rate: 0.0,
                },
                ml_metrics: MlMetrics {
                    total_predictions: 0,
                    average_inference_time: Duration::from_secs(0),
                    last_inference_time: Duration::from_secs(0),
                    prediction_accuracy: 0.0,
                    model_version: "1.0.0".to_string(),
                },
            }),
            trade_start_times: RwLock::new(HashMap::new()),
            api_start_times: RwLock::new(HashMap::new()),
            db_start_times: RwLock::new(HashMap::new()),
            ml_start_times: RwLock::new(HashMap::new()),
        }
    }

    // Trade metrics
    pub async fn start_trade(&self, trade_id: String) {
        self.trade_start_times.write().await.insert(trade_id, Instant::now());
    }

    pub async fn end_trade(&self, trade_id: String, success: bool, profit: f64) -> Result<()> {
        let start_time = self.trade_start_times.write().await.remove(&trade_id)
            .ok_or_else(|| anyhow::anyhow!("Trade ID not found"))?;
        
        let duration = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        let trade_metrics = &mut metrics.trade_metrics;

        trade_metrics.total_trades += 1;
        if success {
            trade_metrics.successful_trades += 1;
            trade_metrics.total_profit += profit;
        } else {
            trade_metrics.failed_trades += 1;
            trade_metrics.total_loss += profit.abs();
        }

        // Update average trade duration
        let total_duration = trade_metrics.average_trade_duration * (trade_metrics.total_trades - 1) as u32;
        trade_metrics.average_trade_duration = (total_duration + duration) / trade_metrics.total_trades as u32;
        
        // Update win rate
        trade_metrics.win_rate = trade_metrics.successful_trades as f64 / trade_metrics.total_trades as f64 * 100.0;

        Ok(())
    }

    // API metrics
    pub async fn start_api_request(&self, request_id: String) {
        self.api_start_times.write().await.insert(request_id, Instant::now());
    }

    pub async fn end_api_request(&self, request_id: String, success: bool) -> Result<()> {
        let start_time = self.api_start_times.write().await.remove(&request_id)
            .ok_or_else(|| anyhow::anyhow!("Request ID not found"))?;
        
        let duration = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        let api_metrics = &mut metrics.api_metrics;

        api_metrics.total_requests += 1;
        if !success {
            api_metrics.failed_requests += 1;
        }

        // Update average response time
        let total_time = api_metrics.average_response_time * (api_metrics.total_requests - 1) as u32;
        api_metrics.average_response_time = (total_time + duration) / api_metrics.total_requests as u32;
        api_metrics.last_response_time = duration;
        
        // Update error rate
        api_metrics.error_rate = api_metrics.failed_requests as f64 / api_metrics.total_requests as f64 * 100.0;

        Ok(())
    }

    // Database metrics
    pub async fn start_db_query(&self, query_id: String) {
        self.db_start_times.write().await.insert(query_id, Instant::now());
    }

    pub async fn end_db_query(&self, query_id: String, success: bool, slow: bool) -> Result<()> {
        let start_time = self.db_start_times.write().await.remove(&query_id)
            .ok_or_else(|| anyhow::anyhow!("Query ID not found"))?;
        
        let duration = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        let db_metrics = &mut metrics.db_metrics;

        db_metrics.total_queries += 1;
        if slow {
            db_metrics.slow_queries += 1;
        }

        // Update average query time
        let total_time = db_metrics.average_query_time * (db_metrics.total_queries - 1) as u32;
        db_metrics.average_query_time = (total_time + duration) / db_metrics.total_queries as u32;
        db_metrics.last_query_time = duration;

        Ok(())
    }

    // ML metrics
    pub async fn start_ml_prediction(&self, prediction_id: String) {
        self.ml_start_times.write().await.insert(prediction_id, Instant::now());
    }

    pub async fn end_ml_prediction(&self, prediction_id: String, accuracy: f64) -> Result<()> {
        let start_time = self.ml_start_times.write().await.remove(&prediction_id)
            .ok_or_else(|| anyhow::anyhow!("Prediction ID not found"))?;
        
        let duration = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        let ml_metrics = &mut metrics.ml_metrics;

        ml_metrics.total_predictions += 1;
        
        // Update average inference time
        let total_time = ml_metrics.average_inference_time * (ml_metrics.total_predictions - 1) as u32;
        ml_metrics.average_inference_time = (total_time + duration) / ml_metrics.total_predictions as u32;
        ml_metrics.last_inference_time = duration;
        
        // Update prediction accuracy
        ml_metrics.prediction_accuracy = accuracy;

        Ok(())
    }

    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_performance_monitor() -> Result<()> {
        let monitor = PerformanceMonitor::new();
        
        // Test trade metrics
        monitor.start_trade("trade1".to_string()).await;
        sleep(Duration::from_millis(100)).await;
        monitor.end_trade("trade1".to_string(), true, 100.0).await?;
        
        // Test API metrics
        monitor.start_api_request("api1".to_string()).await;
        sleep(Duration::from_millis(50)).await;
        monitor.end_api_request("api1".to_string(), true).await?;
        
        // Test DB metrics
        monitor.start_db_query("db1".to_string()).await;
        sleep(Duration::from_millis(20)).await;
        monitor.end_db_query("db1".to_string(), true, false).await?;
        
        // Test ML metrics
        monitor.start_ml_prediction("ml1".to_string()).await;
        sleep(Duration::from_millis(30)).await;
        monitor.end_ml_prediction("ml1".to_string(), 0.95).await?;
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.trade_metrics.total_trades, 1);
        assert_eq!(metrics.api_metrics.total_requests, 1);
        assert_eq!(metrics.db_metrics.total_queries, 1);
        assert_eq!(metrics.ml_metrics.total_predictions, 1);
        
        Ok(())
    }
} 
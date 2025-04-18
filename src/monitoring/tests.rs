use super::*;
use std::time::Duration;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[tokio::test]
async fn test_threshold_manager() {
    let config = ThresholdConfig {
        system_thresholds: SystemThresholds {
            cpu_usage: 90.0,
            memory_usage: 90.0,
            disk_usage: 90.0,
            error_rate: 10.0,
            api_timeout: Duration::from_secs(5),
            db_timeout: Duration::from_secs(5),
        },
        performance_thresholds: PerformanceThresholds {
            api_error_rate: 5.0,
            db_error_rate: 5.0,
            api_response_time: Duration::from_secs(1),
            db_query_time: Duration::from_secs(1),
            ml_inference_time: Duration::from_secs(1),
        },
        trade_thresholds: TradeThresholds {
            win_rate: 50.0,
            max_drawdown: 20.0,
            max_position_size: 1000.0,
            min_profit_per_trade: 10.0,
            max_loss_per_trade: 100.0,
            daily_loss_limit: 1000.0,
        },
        notification_settings: NotificationSettings {
            alert_cooldown: Duration::from_secs(300),
            max_alerts_per_hour: 12,
            alert_priority: HashMap::new(),
        },
    };

    let manager = ThresholdManager::new(config);
    
    // Test system thresholds
    let health_metrics = HealthMetrics {
        cpu_usage: 95.0,
        memory_usage: 85.0,
        disk_usage: 75.0,
        error_rate: 15.0,
        api_status: true,
        db_status: true,
    };
    
    let alerts = manager.check_system_thresholds(&health_metrics);
    assert!(alerts.contains(&"High CPU usage detected".to_string()));
    assert!(alerts.contains(&"High error rate detected".to_string()));
    
    // Test performance thresholds
    let perf_metrics = PerformanceMetrics {
        api_metrics: ApiMetrics {
            error_rate: 6.0,
            average_response_time: Duration::from_secs(2),
        },
        db_metrics: DbMetrics {
            error_rate: 4.0,
            average_query_time: Duration::from_secs(0.5),
        },
        ml_metrics: MlMetrics {
            average_inference_time: Duration::from_secs(0.5),
        },
        trade_metrics: TradeMetrics {
            win_rate: 45.0,
            total_profit: 1000.0,
            total_loss: 500.0,
            position_size: 1200.0,
        },
    };
    
    let alerts = manager.check_performance_thresholds(&perf_metrics);
    assert!(alerts.contains(&"High API error rate detected".to_string()));
    assert!(alerts.contains(&"Slow API response time detected".to_string()));
    
    // Test trade thresholds
    let alerts = manager.check_trade_thresholds(&perf_metrics);
    assert!(alerts.contains(&"Low win rate detected".to_string()));
    assert!(alerts.contains(&"Position size exceeded".to_string()));
}

#[tokio::test]
async fn test_alert_manager() {
    let manager = AlertManager::new(Duration::from_secs(3600));
    
    // Test adding alerts
    let alert = PersistedAlert {
        id: "test_alert".to_string(),
        message: "Test alert".to_string(),
        priority: "high".to_string(),
        category: "system".to_string(),
        timestamp: Utc::now(),
        resolved: false,
        resolved_at: None,
        metadata: HashMap::new(),
    };
    
    manager.add_alert(alert.clone()).await.unwrap();
    
    // Test getting alerts
    let alerts = manager.get_alerts(None, None, None, None, None).await;
    assert_eq!(alerts.len(), 1);
    assert_eq!(alerts[0].id, "test_alert");
    
    // Test resolving alerts
    manager.resolve_alert("test_alert").await.unwrap();
    let alerts = manager.get_alerts(Some(false), None, None, None, None).await;
    assert_eq!(alerts.len(), 0);
    
    // Test alert stats
    let stats = manager.get_stats().await;
    assert_eq!(stats.total_alerts, 1);
    assert_eq!(stats.resolved_alerts, 1);
    assert_eq!(stats.active_alerts, 0);
}

#[tokio::test]
async fn test_dashboard() {
    let config = ThresholdConfig {
        system_thresholds: SystemThresholds {
            cpu_usage: 90.0,
            memory_usage: 90.0,
            disk_usage: 90.0,
            error_rate: 10.0,
            api_timeout: Duration::from_secs(5),
            db_timeout: Duration::from_secs(5),
        },
        performance_thresholds: PerformanceThresholds {
            api_error_rate: 5.0,
            db_error_rate: 5.0,
            api_response_time: Duration::from_secs(1),
            db_query_time: Duration::from_secs(1),
            ml_inference_time: Duration::from_secs(1),
        },
        trade_thresholds: TradeThresholds {
            win_rate: 50.0,
            max_drawdown: 20.0,
            max_position_size: 1000.0,
            min_profit_per_trade: 10.0,
            max_loss_per_trade: 100.0,
            daily_loss_limit: 1000.0,
        },
        notification_settings: NotificationSettings {
            alert_cooldown: Duration::from_secs(300),
            max_alerts_per_hour: 12,
            alert_priority: HashMap::new(),
        },
    };
    
    let dashboard = Dashboard::new(config, Duration::from_secs(1), Duration::from_secs(3600));
    
    // Test updating metrics
    let health_metrics = HealthMetrics {
        cpu_usage: 85.0,
        memory_usage: 75.0,
        disk_usage: 65.0,
        error_rate: 5.0,
        api_status: true,
        db_status: true,
    };
    
    let perf_metrics = PerformanceMetrics {
        api_metrics: ApiMetrics {
            error_rate: 3.0,
            average_response_time: Duration::from_secs(0.5),
        },
        db_metrics: DbMetrics {
            error_rate: 2.0,
            average_query_time: Duration::from_secs(0.3),
        },
        ml_metrics: MlMetrics {
            average_inference_time: Duration::from_secs(0.4),
        },
        trade_metrics: TradeMetrics {
            win_rate: 55.0,
            total_profit: 1500.0,
            total_loss: 500.0,
            position_size: 800.0,
        },
    };
    
    dashboard.update_metrics(&health_metrics, &perf_metrics).await;
    
    // Test getting metrics
    let metrics = dashboard.get_metrics().await;
    assert_eq!(metrics.system_health.cpu_usage, 85.0);
    assert_eq!(metrics.performance.api_error_rate, 3.0);
    assert_eq!(metrics.trading.win_rate, 55.0);
    
    // Test historical metrics
    let historical = dashboard.get_historical_metrics(None, None).await;
    assert_eq!(historical.len(), 1);
    
    // Test alert resolution
    if let Some(alert) = metrics.alerts.first() {
        dashboard.resolve_alert(&alert.id).await;
        let metrics = dashboard.get_metrics().await;
        assert!(metrics.alerts.iter().all(|a| a.id != alert.id));
    }
} 
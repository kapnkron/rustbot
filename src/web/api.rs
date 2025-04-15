use crate::monitoring::dashboard::Dashboard;
use crate::security::SecurityManager;
use actix_web::{web, Responder, HttpResponse, HttpRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use log::{info, error};

#[derive(Debug, Serialize, Deserialize)]
pub struct TradeRequest {
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TradeResponse {
    pub order_id: String,
    pub status: String,
    pub filled_quantity: f64,
    pub average_price: f64,
}

pub struct ApiHandler {
    dashboard: Arc<Dashboard>,
    security: Arc<SecurityManager>,
}

impl ApiHandler {
    pub fn new(dashboard: Arc<Dashboard>, security: Arc<SecurityManager>) -> Self {
        Self { dashboard, security }
    }

    pub async fn configure_routes(cfg: &mut web::ServiceConfig) {
        cfg.service(
            web::scope("/api")
                .route("/health", web::get().to(Self::health_check))
                .route("/metrics", web::get().to(Self::get_metrics))
                .route("/alerts", web::get().to(Self::get_alerts))
                .route("/alerts/{id}/resolve", web::post().to(Self::resolve_alert))
                .route("/trades", web::post().to(Self::place_trade))
                .route("/trades/active", web::get().to(Self::get_active_trades))
                .route("/trades/history", web::get().to(Self::get_trade_history))
                .route("/positions", web::get().to(Self::get_positions))
                .route("/balance", web::get().to(Self::get_balance))
                .route("/config", web::get().to(Self::get_config))
                .route("/config", web::put().to(Self::update_config))
        );
    }

    async fn health_check() -> impl Responder {
        HttpResponse::Ok().json(serde_json::json!({
            "status": "ok",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    }

    async fn get_metrics(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        match dashboard.get_metrics().await {
            Ok(metrics) => HttpResponse::Ok().json(metrics),
            Err(e) => {
                error!("Failed to get metrics: {}", e);
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Failed to get metrics"
                }))
            }
        }
    }

    async fn get_alerts(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        match dashboard.get_metrics().await {
            Ok(metrics) => HttpResponse::Ok().json(metrics.alerts),
            Err(e) => {
                error!("Failed to get alerts: {}", e);
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Failed to get alerts"
                }))
            }
        }
    }

    async fn resolve_alert(
        dashboard: web::Data<Arc<Dashboard>>,
        alert_id: web::Path<String>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        match dashboard.resolve_alert(&alert_id).await {
            Ok(_) => HttpResponse::Ok().json(serde_json::json!({
                "status": "ok",
                "message": "Alert resolved"
            })),
            Err(e) => {
                error!("Failed to resolve alert: {}", e);
                HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Failed to resolve alert"
                }))
            }
        }
    }

    async fn place_trade(
        dashboard: web::Data<Arc<Dashboard>>,
        trade: web::Json<TradeRequest>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement trade execution logic
        HttpResponse::Ok().json(TradeResponse {
            order_id: "12345".to_string(),
            status: "filled".to_string(),
            filled_quantity: trade.quantity,
            average_price: trade.price.unwrap_or(0.0),
        })
    }

    async fn get_active_trades(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement active trades retrieval
        HttpResponse::Ok().json(vec![])
    }

    async fn get_trade_history(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement trade history retrieval
        HttpResponse::Ok().json(vec![])
    }

    async fn get_positions(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement positions retrieval
        HttpResponse::Ok().json(vec![])
    }

    async fn get_balance(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement balance retrieval
        HttpResponse::Ok().json(serde_json::json!({
            "total_balance": 0.0,
            "available_balance": 0.0,
            "unrealized_pnl": 0.0
        }))
    }

    async fn get_config(
        dashboard: web::Data<Arc<Dashboard>>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement config retrieval
        HttpResponse::Ok().json(serde_json::json!({}))
    }

    async fn update_config(
        dashboard: web::Data<Arc<Dashboard>>,
        config: web::Json<serde_json::Value>,
        req: HttpRequest,
    ) -> impl Responder {
        // Verify API key
        if let Some(api_key) = req.headers().get("X-API-Key") {
            if !dashboard.security.validate_api_key(api_key.to_str().unwrap()).await {
                return HttpResponse::Unauthorized().json(serde_json::json!({
                    "error": "Invalid API key"
                }));
            }
        }

        // TODO: Implement config update
        HttpResponse::Ok().json(serde_json::json!({
            "status": "ok",
            "message": "Configuration updated"
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::test;

    #[actix_rt::test]
    async fn test_api_endpoints() {
        // TODO: Implement API endpoint tests
    }
} 
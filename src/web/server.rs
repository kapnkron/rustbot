use crate::monitoring::dashboard::Dashboard;
use crate::security::SecurityManager;
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use actix_cors::Cors;
use std::sync::Arc;
use log::{info, error};

pub struct WebServer {
    dashboard: Arc<Dashboard>,
    security: Arc<SecurityManager>,
}

impl WebServer {
    pub fn new(dashboard: Arc<Dashboard>, security: Arc<SecurityManager>) -> Self {
        Self { dashboard, security }
    }

    pub async fn start(&self, host: &str, port: u16) -> std::io::Result<()> {
        info!("Starting web server on {}:{}", host, port);

        HttpServer::new(move || {
            let cors = Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600);

            App::new()
                .wrap(cors)
                .app_data(web::Data::new(self.dashboard.clone()))
                .app_data(web::Data::new(self.security.clone()))
                .route("/health", web::get().to(Self::health_check))
                .route("/metrics", web::get().to(Self::get_metrics))
                .route("/alerts", web::get().to(Self::get_alerts))
                .route("/alerts/{id}/resolve", web::post().to(Self::resolve_alert))
        })
        .bind((host, port))?
        .run()
        .await
    }

    async fn health_check() -> impl Responder {
        HttpResponse::Ok().json(serde_json::json!({
            "status": "ok",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    }

    async fn get_metrics(dashboard: web::Data<Arc<Dashboard>>) -> impl Responder {
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

    async fn get_alerts(dashboard: web::Data<Arc<Dashboard>>) -> impl Responder {
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
        alert_id: web::Path<String>
    ) -> impl Responder {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_web_server_creation() {
        let config = ThresholdConfig {
            // ... threshold configuration ...
        };
        let dashboard = Arc::new(Dashboard::new(config, Duration::from_secs(1)));
        let security = Arc::new(SecurityManager::new(SecurityConfig {
            // ... security configuration ...
        }).unwrap());
        
        let server = WebServer::new(dashboard, security);
        assert!(server.start("127.0.0.1", 8080).await.is_ok());
    }
} 
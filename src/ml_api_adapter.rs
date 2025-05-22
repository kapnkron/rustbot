use serde::{Deserialize, Serialize};
// use std::collections::HashMap; // Not strictly needed for these structs

// Import the crate's error type
use crate::error::{Error, Result}; // Assuming src/lib.rs or src/main.rs declares 'pub mod error;' and re-exports Result

// Corresponds to Python's PredictRequest
#[derive(Debug, Serialize, Default)]
pub struct PredictRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<String>>,
    // The Pydantic model has 'force_feature_recalculation' (snake_case)
    // By default, serde serializes Rust snake_case to JSON snake_case.
    // If the Python API *expects* camelCase (e.g. forceFeatureRecalculation), 
    // then #[serde(rename = "forceFeatureRecalculation")] would be needed.
    // Assuming Python API expects snake_case for this field as per Pydantic default.
    pub force_feature_recalculation: bool,
}

// Corresponds to Python's Signal
#[derive(Debug, Deserialize)]
pub struct Signal {
    // Serde by default expects JSON keys to match Rust field names (snake_case).
    // If Python API returns camelCase (e.g. tokenAddress), then #[serde(rename = "tokenAddress")] is needed.
    // Assuming Python API returns snake_case as per Pydantic default.
    pub token_address: Option<String>,
    pub current_price_actual: Option<f64>,
    pub predicted_next_price: f64,
    pub signal_strength: Option<f64>,
    pub raw_prediction: Option<Vec<f64>>,
    pub predicted_action_label: Option<i32>,
    pub action: Option<String>,
}

// Corresponds to Python's PredictionMetadata
#[derive(Debug, Deserialize)]
pub struct PredictionMetadata {
    pub model_version: Option<String>,
    pub feature_set_version: Option<String>,
    pub prediction_time_ms: Option<i64>,
}

// Corresponds to Python's PredictResponse
#[derive(Debug, Deserialize)]
pub struct PredictResponse {
    pub signals: Vec<Signal>,
    pub metadata: PredictionMetadata,
    pub request_timestamp_utc: String,
    pub response_timestamp_utc: String,
}

// --- MlApiClient Struct and Implementation --- 

/// A client for interacting with the ML API.
#[derive(Debug)] // reqwest::Client is not Clone, so MlApiClient can't be Clone by default
pub struct MlApiClient {
    client: reqwest::Client,
    base_url: String,
}

impl MlApiClient {
    /// Creates a new `MlApiClient`.
    ///
    /// # Arguments
    /// * `base_url` - The base URL for the ML API (e.g., "http://localhost:8000").
    pub fn new(base_url: String) -> Self {
        MlApiClient {
            client: reqwest::Client::new(), // Creates a new client with default settings
            base_url,
        }
    }

    /// Calls the ML API's /predict endpoint.
    pub async fn get_predictions(
        &self, // Takes &self to use its fields
        request_data: &PredictRequest
    ) -> Result<PredictResponse> { // Uses the crate::error::Result
        let url = format!("{}/predict", self.base_url);

        log::debug!("Sending prediction request to {}: {:?}", url, request_data);

        let response = self.client // Use self.client
            .post(&url)
            .json(request_data) // Serializes PredictRequest to JSON
            .send()
            .await
            .map_err(|e| {
                log::error!("Request to ML API /predict failed: {}", e);
                Error::ApiConnectionFailed(format!("Failed to connect to ML API /predict: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
            log::error!("ML API /predict returned error status {}: {}", status, error_text);
            return Err(Error::ApiError(format!(
                "ML API /predict failed with status {}: {}", status, error_text
            )));
        }

        match response.json::<PredictResponse>().await {
            Ok(predict_response) => {
                log::debug!("Received prediction response: {:?}", predict_response);
                Ok(predict_response)
            }
            Err(e) => {
                log::error!("Failed to deserialize /predict response: {}", e);
                Err(Error::ApiInvalidFormat(format!("Failed to parse /predict response: {}", e)))
            }
        }
    }
    
    // TODO: Add methods for /data/refresh_pipeline and /model/retrain later
}

// --- Tests --- (Adapted to use MlApiClient)

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_predict_request_serialization() {
        let req = PredictRequest {
            tokens: Some(vec!["token1".to_string(), "token2".to_string()]),
            force_feature_recalculation: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tokens\":[\"token1\",\"token2\"]"));
        assert!(json.contains("\"force_feature_recalculation\":true"));

        let req_no_tokens = PredictRequest {
            tokens: None,
            force_feature_recalculation: false,
        };
        let json_no_tokens = serde_json::to_string(&req_no_tokens).unwrap();
        assert!(!json_no_tokens.contains("\"tokens\":"));
        assert!(json_no_tokens.contains("\"force_feature_recalculation\":false"));

        let req_default = PredictRequest::default();
        let json_default = serde_json::to_string(&req_default).unwrap();
        assert!(!json_default.contains("\"tokens\":"));
        assert!(json_default.contains("\"force_feature_recalculation\":false"));
    }

    #[tokio::test]
    #[ignore] // Ignored because it requires a running mock server or the actual API
    async fn test_get_predictions_with_mock_client() {
        // This test would ideally use a mock HTTP server like wiremock.
        // For now, it mainly tests the deserialization logic if we could mock the response.

        // let ml_api_client = MlApiClient::new("http://mock-server.local".to_string());
        // In a real test with a mock server, you'd get its URI for the base_url.

        let request_data = PredictRequest {
            tokens: Some(vec!["some_token_address".to_string()]),
            force_feature_recalculation: false,
        };

        let mock_response_json = r#"
        {
            "signals": [
                {
                    "token_address": "some_token_address",
                    "current_price_actual": 100.50,
                    "predicted_next_price": 101.75,
                    "signal_strength": 0.8,
                    "raw_prediction": [0.75],
                    "predicted_action_label": 0,
                    "action": "Buy"
                }
            ],
            "metadata": {
                "model_version": "mv1.0",
                "feature_set_version": "fv1.1",
                "prediction_time_ms": 55
            },
            "request_timestamp_utc": "2024-01-01T12:00:00Z",
            "response_timestamp_utc": "2024-01-01T12:00:01Z"
        }
        "#;
        
        // To truly test `ml_api_client.get_predictions()`, you would need to:
        // 1. Set up a mock HTTP server (e.g., using `wiremock`).
        // 2. Configure the mock server to respond to POST /predict with `mock_response_json`.
        // 3. Create `ml_api_client` with the mock server's URL.
        // 4. Call `ml_api_client.get_predictions(&request_data).await`.
        // 5. Assert the `Result` is `Ok` and the `PredictResponse` matches expectations.

        // For now, we just re-test the deserialization part.
        let parsed: std::result::Result<PredictResponse, serde_json::Error> = serde_json::from_str(mock_response_json);
        assert!(parsed.is_ok());
        let predict_response = parsed.unwrap();
        assert_eq!(predict_response.signals.len(), 1);
        assert_eq!(predict_response.signals[0].token_address.as_deref(), Some("some_token_address"));
    }
} 
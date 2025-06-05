# Project Beatrice: Solana Trading Bot - Status and Next Steps

**Date:** May 19, 2025 (Simulated)

## 🎯 Current Goal:
Integrate a Python-based Machine Learning (ML) pipeline with a Rust core for a Solana trading bot, enabling the Rust core to fetch predictions from a Python API.

## ✅ Recent Accomplishments (Focus: Fixing ML Prediction Pipeline):

The primary focus has been on resolving a critical "feature mismatch" error that occurred when the Rust application called the Python API's `/predict` endpoint. The Python ML pipeline (data fetching, feature engineering, training, prediction) was returning errors because the number of features in the prediction request didn't match the model's expectations.

1.  **Identified Root Cause:** The prediction script (`ml_pipeline/prediction.py`) was dynamically inferring numerical feature columns from the input data. This was unreliable and led to inconsistencies with the features used during model training (specifically for `scaler_X_trading.pkl`).
2.  **Implemented Robust Feature Set Management:**
    *   **`ml_pipeline/training.py` Modifications:**
        *   Now explicitly identifies and saves the list of numerical feature column names used during each training run into a `numerical_columns.json` file.
        *   Saves all artifacts (model, scalers, asset encoder, `numerical_columns.json`) into a timestamped subdirectory within `models/neural_network/` for better versioning and organization.
    *   **`ml_pipeline/prediction.py` Modifications:**
        *   Updated to load the `numerical_columns.json` file.
        *   Uses this explicit list of numerical features to prepare data for prediction, ensuring an exact match with the training conditions.
        *   Includes enhanced checks and logging for artifact loading and feature validation.
        *   The standalone test block (`if __name__ == '__main__'`) was improved to create dummy artifacts for more robust local testing.
3.  **Python API Server (`api_server.py`):**
    *   The `/predict` endpoint now correctly utilizes the updated `ml_pipeline/prediction.py` logic.
4.  **Rust Integration (`src/ml_api_adapter.rs`, `src/main.rs`):**
    *   The `MlApiClient` in Rust successfully calls the Python `/predict` endpoint.
    *   The system now successfully makes an end-to-end prediction: Rust client -> Python API -> ML model -> Rust client.
    *   The Python API returns a `200 OK` status, and the Rust client receives the prediction data.
5.  **Testing & Validation:**
    *   Manually ran the training pipeline (`python -m ml_pipeline.training`) to generate a consistent set of artifacts.
    *   Manually copied these artifacts to the project root (for `model_trading.pt`, `scaler_X_trading.pkl`, etc.) for the API server to use.
    *   Confirmed successful prediction via `cargo run --bin trading_bot`, observing the API's `200 OK` and the Rust client printing the received prediction data.

**The critical "feature mismatch" bug is now resolved.**

### Solana Historical Data Fetcher (`fetch_ohlcv_rolling.py`):
*   **Goal:** Fetch historical trade data for SOL-based pairs (initially Openbook, now also other DEXs like Orca using Helius API) to generate 5-minute OHLCV candles for ML model training and the orchestration script (`update_and_prepare_training_data.py`).
*   **Progress:**
    *   Switched from free RPC to Helius paid tier for increased API limits.
    *   Refactored to use Helius Enhanced Transactions API (`/v0/addresses/{address}/transactions`).
    *   Updated to select the most liquid pool (>$10k, SOL/USDC quote) in `fetch_pool_addresses.py`.
    *   Implemented logic for initial 30-day data fetch for new pools and incremental updates for existing ones in `fetch_ohlcv_rolling.py`.
    *   Added debug `print` statements to `fetch_ohlcv_rolling.py` to investigate data fetching issues.
*   **Current Status & Issue:**
    *   Debugging `fetch_ohlcv_rolling.py` revealed that Helius API calls are failing with the error: `{"error":"invalid query parameter limit"}`.
    *   This occurs for the `https://api.helius.xyz/v0/addresses/{address}/transactions` endpoint.
    *   The `limit` parameter is documented as valid. The issue might be a subtle type mismatch for `limit` or a conflict with pagination parameters if `before` is used in the initial request. Helius documentation suggests using `before` and `until` for pagination, along with `limit`.
*   **File Cleanup:**
    *   All CSVs in `data/raw/` and `data/features/` have been deleted as part of streamlining data processing. Signature list CSVs are intended to be preserved.
*   **Known Issues (previously):**
    *   The script previously overwrote existing OHLCV CSV files. This was addressed by implementing incremental updates based on the last timestamp in existing files.

## 🚀 Immediate Next Steps:

1.  **Resolve Helius API "invalid query parameter limit" Error:**
    *   **File:** `scripts/fetch_ohlcv_rolling.py`
    *   **Task:**
        *   Modify the Helius API call logic.
        *   Use `until` instead of `after` for the initial time bound.
        *   Fetch in a loop, using the `before` signature from the last transaction of the previous batch for pagination, until no more transactions are returned or the `start_time` is passed.
        *   Ensure the `limit` parameter is correctly passed as an integer.
        *   Verify that this resolves the error and allows successful data fetching.

2.  **Integrate Predictions into Rust Bot Logic:**
    *   **File:** `src/main.rs`
    *   **Task:**
        *   Remove the temporary test call block for `ml_api_client.get_predictions()`.
        *   Uncomment and refactor the main trading loop and `TradingBot` initialization/usage.
        *   Replace any old `PythonPredictor` logic with calls to `ml_api_client.get_predictions()` to fetch predictions as needed for trading decisions.
        *   Ensure the `PredictResponse` data is correctly utilized by the bot.

3.  **Automate ML Artifact Management:**
    *   **Files:** `ml_pipeline/training.py`, `ml_pipeline/prediction.py`, `api_server.py`.
    *   **Task:**
        *   Currently, artifacts are manually copied from the timestamped training output directory (e.g., `models/neural_network/YYYYMMDD_HHMMSS/`) to the project root and renamed. This needs automation.
        *   **Option A (Symlink):** `training.py` (or the `/model/retrain` API endpoint) could create/update a symlink (e.g., `models/neural_network/latest`) pointing to the latest successfully trained artifact directory. `prediction.py` would then load artifacts from this `latest` symlink.
        *   **Option B (Manifest File):** `training.py` could generate a `latest_model_info.json` pointing to the paths of the current production artifacts. `prediction.py` would read this manifest.
        *   The goal is for `api_server.py` to automatically use the most recent, validated set of artifacts without manual intervention after retraining.

4.  **Code Cleanup - Rust:**
    *   **Files:** Entire Rust codebase (`src/`).
    *   **Task:**
        *   Address the numerous `#[warn(unused_imports)]` and `#[warn(dead_code)]` warnings.
        *   Run `cargo fix --lib -p trading_bot --allow-dirty` and `cargo fix --bin "trading_bot" --allow-dirty` (use `--allow-dirty` or commit changes first).
        *   Manually review and remove any remaining unused code to improve maintainability.

5.  **Code Cleanup - Python (Minor):**
    *   **File:** `api_server.py` (and potentially other Python files).
    *   **Task:** Address `DeprecationWarning: datetime.datetime.utcnow() is deprecated`. Replace `datetime.utcnow()` with `datetime.now(datetime.UTC)`.

6.  **Verify Full Data Pipeline Post Helius Fix:**
    *   **Files:** `update_and_prepare_training_data.py`, `scripts/fetch_pool_addresses.py`, `scripts/fetch_ohlcv_rolling.py`, `scripts/prepare_training_data.py`.
    *   **Task:** After resolving the Helius API issue, run the orchestrator script (`update_and_prepare_training_data.py`) to ensure the entire data pipeline (pool discovery, OHLCV fetching, training data preparation including the 6-month initial data logic) works end-to-end.
    *   Monitor logs (`data_fetcher.log`) and output files in `data/processed/` and `data/features/`.

## ⏳ Future/Blocked Tasks (Previously Discussed - Lower Priority for now):

*   Full implementation of `TradingBot` logic beyond basic prediction fetching.
*   Telegram bot integration and dashboard functionality (currently mostly commented out).
*   Advanced error handling, monitoring, and alerting.
*   Database integration for trade logging, performance tracking, etc.

This document should provide a clear overview for handoff. The immediate priority is to integrate the now-working prediction mechanism into the Rust bot's core decision-making loop and then streamline artifact management. 
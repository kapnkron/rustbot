import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uvicorn # For running the server

# Add the parent directory to sys.path to allow imports from ml_pipeline
# This assumes api_server.py is in the root and ml_pipeline is a subdirectory.
# Adjust if your structure is different.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'ml_pipeline' is at the same level as where this script might eventually be,
# or that your project root is in PYTHONPATH.
# If api_server.py is in the root, and prediction.py is in ml_pipeline/prediction.py:
project_root = current_dir # Or os.path.dirname(current_dir) if api_server is in a subfolder itself
sys.path.append(project_root) 
# sys.path.append(os.path.join(project_root, "ml_pipeline")) # If modules are directly in ml_pipeline


# Import the refactored prediction function
# Ensure your PYTHONPATH is set up correctly or adjust path as needed
try:
    from ml_pipeline.prediction import get_predictions_for_api
    # We'll import functions for other endpoints later
    # from ml_pipeline.data_pipeline import run_full_data_pipeline
    from ml_pipeline.feature_engineering import engineer_features
    from ml_pipeline.training import run_training_pipeline
except ImportError as e:
    print(f"Error importing ML modules: {e}")
    print(f"Ensure ml_pipeline is in PYTHONPATH or sys.path is configured correctly.")
    print(f"Current sys.path: {sys.path}")
    # Exit if core components can't be imported, or handle more gracefully
    sys.exit(1)

# Import the train_and_backtest function
sys.path.append(os.path.join(current_dir, 'scripts'))
try:
    from scripts.train_and_backtest import train_and_backtest
except ImportError as e:
    print(f"Error importing train_and_backtest: {e}")
    train_and_backtest = None

app = FastAPI(
    title="Trading Bot ML API",
    description="API for interacting with the Python ML module for trading predictions and operations.",
    version="0.1.0"
)

# --- Pydantic Models for Request and Response Schemas ---

class PredictRequest(BaseModel):
    tokens: Optional[List[str]] = Field(None, description="List of token addresses to predict for. If empty or null, predicts for all relevant tokens.")
    force_feature_recalculation: bool = Field(False, description="If true, forces recalculation of features before prediction.")

class Signal(BaseModel):
    token_address: Optional[str]
    current_price_actual: Optional[float]
    predicted_next_price: float # This will now be set to current_price_actual by prediction.py
    signal_strength: Optional[float] = None # This will be the confidence (max probability)
    raw_prediction: Optional[List[float]] = None # This will be the list of class probabilities [P(Buy), P(Sell), P(Hold)]
    predicted_action_label: Optional[int] = None # The predicted class index (0:Buy, 1:Sell, 2:Hold)
    action: Optional[str] = None # The string representation of the action ("Buy", "Sell", "Hold")
    # timestamp_utc will be added by the endpoint

class PredictionMetadata(BaseModel):
    model_version: Optional[str]
    feature_set_version: Optional[str]
    prediction_time_ms: Optional[int] # Time taken within the python script
    # request_timestamp_utc will be added by the endpoint

class PredictResponse(BaseModel):
    signals: List[Signal]
    metadata: PredictionMetadata
    request_timestamp_utc: str # Timestamp when the API received the request
    response_timestamp_utc: str # Timestamp when the API is sending the response

class StatusResponse(BaseModel):
    service_status: str
    # last_data_update_utc: Optional[str] = None # Implement later
    # last_model_retrain_utc: Optional[str] = None # Implement later
    # current_model_version: Optional[str] = None # Implement later
    # active_tasks: List[str] = [] # Implement later
    server_timestamp_utc: str

class ActionInitiatedResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None # For async tasks later
    timestamp_utc: str

class DataRefreshDetail(BaseModel):
    input_path: str
    output_path: str
    status: str
    message: str
    initial_shape: Optional[tuple[int, int]] = None
    final_shape: Optional[tuple[int, int]] = None
    num_features_generated: Optional[int] = None

class DataRefreshResponse(BaseModel):
    overall_status: str
    message: str
    details: List[DataRefreshDetail]
    timestamp_utc: str

class TrainAndBacktestRequest(BaseModel):
    train_path: Optional[str] = 'data/processed/train_data.csv'
    test_path: Optional[str] = 'data/processed/test_data.csv'
    model_dir: Optional[str] = 'models'

class TrainAndBacktestResponse(BaseModel):
    train_path: str
    test_path: str
    model_dir: str
    metrics: dict

# --- Pydantic Models for Retraining ---
class RetrainRequest(BaseModel):
    data_path: Optional[str] = Field(None, description="Path to the training features CSV file. Uses default if not provided.")
    model_dir: Optional[str] = Field(None, description="Directory to save the trained model and artifacts. Uses default if not provided.")
    val_size: Optional[float] = Field(None, description="Proportion of data for validation.")
    target_column: Optional[str] = Field(None, description="Target column name for prediction.")
    hidden_dim1: Optional[int] = Field(None, description="Size of the first hidden layer.")
    hidden_dim2: Optional[int] = Field(None, description="Size of the second hidden layer.")
    dropout_rate: Optional[float] = Field(None, description="Dropout rate for the model.")
    epochs: Optional[int] = Field(None, description="Number of training epochs.")
    batch_size: Optional[int] = Field(None, description="Batch size for training.")
    learning_rate: Optional[float] = Field(None, description="Learning rate for the optimizer.")
    patience: Optional[int] = Field(None, description="Patience for early stopping.")
    use_robust_scaler_y: Optional[bool] = Field(None, description="Whether to use RobustScaler for the target variable.")
    random_state: Optional[int] = Field(None, description="Random state for reproducibility.")

class RetrainResponse(BaseModel):
    status: str
    message: str
    run_timestamp: str
    model_path: Optional[str] = None
    scaler_X_path: Optional[str] = None
    scaler_y_path: Optional[str] = None
    asset_encoder_path: Optional[str] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    input_data_path: Optional[str] = None
    model_architecture: Optional[str] = None

# --- API Endpoints ---

@app.post("/predict", response_model=PredictResponse)
async def predict_signals(request_data: PredictRequest):
    """
    Generates trading predictions based on the latest available data and model.
    """
    request_time = datetime.now(timezone.utc)
    print(f"Received /predict request at {request_time.isoformat()}Z with tokens: {request_data.tokens}")

    try:
        # Call the refactored prediction logic
        # For now, assume get_predictions_for_api handles its own timing for 'prediction_time_ms'
        # and can determine model/feature versions.
        # We pass the path to the features file that `get_predictions_for_api` expects.
        # This file should be the output of your data pipeline (e.g., `features_test_ohlcv.csv`)
        
        # Start timing for prediction_time_ms
        processing_start_time = datetime.now(timezone.utc)

        prediction_results = get_predictions_for_api(
            tokens_filter=request_data.tokens
            # features_file_path will use default from prediction.py or you can pass it here
            # force_feature_recalculation is not yet used by get_predictions_for_api, needs wiring up
        )
        
        processing_end_time = datetime.now(timezone.utc)
        prediction_duration_ms = int((processing_end_time - processing_start_time).total_seconds() * 1000)

        if isinstance(prediction_results, dict) and "error" in prediction_results:
            print(f"Error from prediction module: {prediction_results['error']}")
            # You might want to map specific errors from your module to HTTP status codes
            raise HTTPException(status_code=500, detail=prediction_results['error'])

        ACTION_MAP = {0: "Buy", 1: "Sell", 2: "Hold"} # Define this map for use

        signals_for_response = []
        for res_dict in prediction_results: # res_dict is the dict from prediction.py
            # Populate the action string based on the predicted_action_label
            action_label = res_dict.get("predicted_action_label")
            if action_label is not None:
                res_dict["action"] = ACTION_MAP.get(action_label, "Unknown")
            else:
                res_dict["action"] = "Unknown" # Default if label is missing for some reason
            
            signals_for_response.append(Signal(**res_dict)) # Unpack dict into Pydantic model

        # TODO: Populate these from the actual prediction process if available
        metadata = PredictionMetadata(
            model_version="model_v_placeholder", # Get from loaded model or config
            feature_set_version="features_v_placeholder", # Get from data source
            prediction_time_ms=prediction_duration_ms
        )
        
        response_time = datetime.now(timezone.utc)
        return PredictResponse(
            signals=signals_for_response,
            metadata=metadata,
            request_timestamp_utc=request_time.isoformat() + "Z",
            response_timestamp_utc=response_time.isoformat() + "Z"
        )

    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except FileNotFoundError as e:
        print(f"FileNotFoundError in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"A required ML resource file was not found: {e.filename}")
    except Exception as e:
        import traceback
        print("Unhandled error in /predict endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred on the server: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def get_server_status():
    """
    Provides the current operational status of the ML API server.
    """
    # Basic status for now.
    # Later, this can check model file dates, data file dates, etc.
    return StatusResponse(
        service_status="ok",
        server_timestamp_utc=datetime.now(timezone.utc).isoformat() + "Z"
    )

# --- Placeholder Endpoints for Data Pipeline & Training ---

@app.post("/data/refresh_pipeline", response_model=DataRefreshResponse)
async def trigger_data_refresh_pipeline():
    """
    Initiates the feature engineering process for train and test datasets.
    This endpoint is synchronous for now.
    """
    request_time = datetime.now(timezone.utc)
    print(f"Received /data/refresh_pipeline request at {request_time.isoformat()}Z")

    # Define input and output paths similar to the standalone script
    # These paths are relative to the project root where api_server.py is expected to run.
    # Adjust if your API server runs from a different location relative to 'data/'
    paths_to_process = [
        {"input": "data/processed/train_data.csv", "output": "data/features_train_ohlcv.csv", "dataset_type": "train"},
        {"input": "data/processed/test_data.csv", "output": "data/features_test_ohlcv.csv", "dataset_type": "test"}
    ]

    results_details = []
    all_successful = True
    error_messages = []

    for paths in paths_to_process:
        input_path = paths["input"]
        output_path = paths["output"]
        dataset_type = paths["dataset_type"]
        
        print(f"Processing {dataset_type} data: {input_path} -> {output_path}")
        try:
            # Ensure the directory for the output file exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
        
            result = engineer_features(input_csv_path=input_path, output_csv_path=output_path)
            
            detail = DataRefreshDetail(
                input_path=result.get("input_path", input_path),
                output_path=result.get("output_path", output_path),
                status=result.get("status", "unknown"),
                message=result.get("message", "No message returned"),
                initial_shape=result.get("initial_shape"),
                final_shape=result.get("final_shape"),
                num_features_generated=result.get("num_features_generated")
            )
            results_details.append(detail)

            if result.get("status") != "success":
                all_successful = False
                error_messages.append(f"{dataset_type.capitalize()} data processing failed: {result.get('message', 'Unknown error')}")
                print(f"Error processing {dataset_type} data: {result.get('message')}")
            else:
                print(f"Successfully processed {dataset_type} data. Output at {output_path}")

        except Exception as e:
            import traceback
            print(f"Unhandled exception while processing {dataset_type} data ({input_path}): {e}")
            traceback.print_exc()
            all_successful = False
            error_messages.append(f"Exception processing {dataset_type} data: {str(e)}")
            results_details.append(DataRefreshDetail(
                input_path=input_path,
                output_path=output_path,
                status="exception",
                message=f"Unhandled exception: {str(e)}"
            ))

    response_time = datetime.now(timezone.utc)
    overall_status = "completed" if all_successful else "failed_partial"
    if not results_details: # Should not happen with current logic but good to check
        overall_status = "failed_no_tasks_run"
        
    final_message = "Feature engineering pipeline processed."
    if not all_successful:
        final_message = "Feature engineering pipeline encountered errors: " + "; ".join(error_messages)
    
    print(f"Overall status for /data/refresh_pipeline: {overall_status}")

    return DataRefreshResponse(
        overall_status=overall_status,
        message=final_message,
        details=results_details,
        timestamp_utc=response_time.isoformat() + "Z"
    )

@app.post("/model/retrain", response_model=RetrainResponse)
async def trigger_model_retraining(request: RetrainRequest):
    """
    Initiates the model retraining process using parameters from the request.
    All parameters are optional and will use defaults from run_training_pipeline if not provided.
    """
    request_time_utc = datetime.now(timezone.utc).isoformat() + "Z"
    print(f"Received /model/retrain request at {request_time_utc}")

    # Filter out None values from the request to use defaults in run_training_pipeline
    training_params = {k: v for k, v in request.model_dump().items() if v is not None}
    
    print(f"Calling run_training_pipeline with params: {training_params}")

    if not callable(run_training_pipeline):
        print("Error: run_training_pipeline is not callable. Check import.")
        raise HTTPException(status_code=500, detail="Model training function is not available.")

    try:
        result = run_training_pipeline(**training_params)
        
        # Ensure the response matches the Pydantic model
        # The run_training_pipeline function already returns a dict that should match RetrainResponse fields.
        return RetrainResponse(**result)

    except FileNotFoundError as e:
        print(f"FileNotFoundError during model retraining: {e}")
        raise HTTPException(status_code=404, detail=f"A required file was not found: {e.filename}")
    except ValueError as e: # Catch specific errors like data issues
        print(f"ValueError during model retraining: {e}")
        raise HTTPException(status_code=400, detail=f"Data validation or parameter error: {str(e)}")
    except Exception as e:
        import traceback
        print(f"Unhandled error in /model/retrain endpoint at {datetime.now(timezone.utc).isoformat()}Z:")
        traceback.print_exc()
        # Ensure a RetrainResponse-compatible error structure is returned if possible,
        # or fall back to a generic HTTPException.
        # For now, let's return a generic 500 for unexpected errors.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during model retraining: {str(e)}")

@app.post("/train_and_backtest", response_model=TrainAndBacktestResponse)
async def run_train_and_backtest(request: TrainAndBacktestRequest):
    """
    Trains the model and runs backtesting. Returns training and backtesting results.
    """
    if train_and_backtest is None:
        raise HTTPException(status_code=500, detail="train_and_backtest function is not available.")
    try:
        results = train_and_backtest(
            train_path=request.train_path,
            test_path=request.test_path,
            model_dir=request.model_dir
        )
        return TrainAndBacktestResponse(**results)
    except Exception as e:
        import traceback
        print("Error in /train_and_backtest endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during training and backtesting: {str(e)}")

# --- Main entry point to run the server ---
if __name__ == "__main__":
    print("Starting ML API server...")
    # Uvicorn is an ASGI server, good for FastAPI
    # You can run this script directly: python api_server.py
    # Or using uvicorn from the command line: uvicorn api_server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)


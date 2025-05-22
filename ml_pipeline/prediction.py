import pandas as pd
import numpy as np
import torch
import joblib
import os
from typing import Any, List, Dict, Optional
import json # Added import

# Import Net from the model definition file within the same package
from .your_model_definition_file import Net

# Import specific scaler types if known, otherwise use Any
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# Use the latest symlink for all model artifacts
MODEL_DIR = 'models/neural_network/latest'
MODEL_PATH = os.path.join(MODEL_DIR, 'model_net.pth')
SCALER_X_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
# SCALER_Y_PATH = os.path.join(MODEL_DIR, 'scaler_y.pkl') # REMOVED: No scaler_y for classification
ASSET_ENCODER_PATH = os.path.join(MODEL_DIR, 'asset_encoder.pkl')
NUMERICAL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'numerical_columns.json')
DEFAULT_FEATURES_PATH = 'data/features_test_ohlcv.csv' # This should now contain 'action_label' for consistency if used for eval

# MIN_RAW_SCALED_PRED_CLIP = -1.15800476 # REMOVED: Clipping was for regression model's scaled output
# MAX_RAW_SCALED_PRED_CLIP = 2.69177508 # REMOVED

ACTION_MAP = {0: "Buy", 1: "Sell", 2: "Hold"} # Mapping from class index to action string

def load_model_and_scalers(model_path, scaler_x_path, asset_encoder_path, numerical_columns_path):
    """Loads the PyTorch model, scaler_X, asset encoder, and numerical column names."""
    try:
        print(f"Attempting to load scaler_X from absolute path: {os.path.abspath(scaler_x_path)}")
        scaler_X: StandardScaler = joblib.load(scaler_x_path)
        asset_encoder: OneHotEncoder = joblib.load(asset_encoder_path)
        
        with open(numerical_columns_path, 'r') as f:
            numerical_column_names = json.load(f)
        
        num_ohe_features = sum(len(cat) for cat in asset_encoder.categories_)
        expected_combined_features = len(numerical_column_names) + num_ohe_features

        input_dim = expected_combined_features
        if hasattr(scaler_X, 'n_features_in_') and scaler_X.n_features_in_ != expected_combined_features:
            print(f"Warning: scaler_X.n_features_in_ ({scaler_X.n_features_in_}) does not match "
                  f"the sum of loaded numerical_columns ({len(numerical_column_names)}) and OHE features "
                  f"derived from loaded asset_encoder ({num_ohe_features}). Expected sum: {expected_combined_features}.")
            input_dim = scaler_X.n_features_in_ # Prioritize scaler's expectation if mismatch
        elif not hasattr(scaler_X, 'n_features_in_'):
            print(f"Warning: scaler_X does not have n_features_in_. Using sum of numerical & OHE features: {input_dim}")

        model = Net(input_dim=input_dim, output_dim=3) # Ensure output_dim=3 for classification
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        if not list(model.parameters())[0].sum().item() == 0 and not any(torch.isnan(p).any() for p in model.parameters()):
            print("Model state loaded and seems valid (non-zero sum, no NaNs in first parameter).")
        else:
            print("Warning: Model parameters might be all zeros or contain NaNs after loading state_dict. Check model file and training.")

        model.eval()

        # scaler_y: RobustScaler = joblib.load(scaler_y_path) # REMOVED: No scaler_y
        print(f"Model, scaler_X, asset encoder, and numerical columns list loaded. Model input_dim set to {input_dim}.")
        return model, scaler_X, asset_encoder, numerical_column_names # REMOVED scaler_y from return
    except FileNotFoundError as e:
        print(f"Error loading model/scaler/numerical_columns: {e}")
        raise
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred loading model/scalers/numerical_columns: {e}")
        traceback.print_exc()
        raise


def preprocess_input_data(df: pd.DataFrame, scaler_X: StandardScaler, asset_encoder: OneHotEncoder, numerical_feature_cols: List[str]): # Added type hint for df and feature_columns
    """Prepares DataFrame for prediction."""
    if df.empty:
        return np.array([]).reshape(0, scaler_X.n_features_in_ if hasattr(scaler_X, 'n_features_in_') else 0), \
               pd.Series(dtype='object'), \
               pd.Series(dtype='float64')

    df_numerical_features = df[numerical_feature_cols]
    
    # Apply OneHotEncoder to the token_address column
    encoded_assets = asset_encoder.transform(df[['token_address']]) # Must be 2D
    encoded_asset_cols = asset_encoder.get_feature_names_out(['token_address'])
    encoded_assets_df = pd.DataFrame(encoded_assets, columns=encoded_asset_cols, index=df.index)
    
    # Concatenate numerical features and OHE token_address features
    X_combined = pd.concat([df_numerical_features, encoded_assets_df], axis=1)

    # Ensure the order of columns in X_combined matches what scaler_X was trained on
    # This is crucial if scaler_X.feature_names_in_ is not None and was used during fitting.
    # For now, we rely on scaler_X.n_features_in_ and assume the concatenation order + numerical_feature_cols is correct.
    # If scaler_X was fit with feature names, X_combined should be reordered to match scaler_X.feature_names_in_

    if X_combined.shape[1] != scaler_X.n_features_in_:
        raise ValueError(
            f"Post-OHE feature count ({X_combined.shape[1]}) does not match scaler_X expected features ({scaler_X.n_features_in_}).\n"
            f"Numerical features considered: {numerical_feature_cols}\n"
            f"OHE columns generated: {list(encoded_asset_cols)}"
        )

    X_imputed = np.nan_to_num(X_combined.values, nan=0.0)
    X_scaled = scaler_X.transform(X_imputed)
    
    return X_scaled, df['token_address'], df['price']


def generate_predictions(
    features_df: pd.DataFrame,
    model: torch.nn.Module,
    scaler_X: StandardScaler, 
    # scaler_y: RobustScaler, # REMOVED: No scaler_y for classification
    asset_encoder: OneHotEncoder,
    numerical_feature_cols: List[str]
):
    """Generates action predictions and confidence for the given features_df."""
    if features_df.empty:
        return []

    X_scaled, token_addresses, actual_current_prices = preprocess_input_data(
        features_df, scaler_X, asset_encoder, numerical_feature_cols 
    )

    if X_scaled.shape[0] == 0 :
        return []

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        log_probs_tensor = model(X_tensor) # Model outputs log probabilities
        probs_tensor = torch.exp(log_probs_tensor) # Convert to probabilities
    
    # Get the predicted class (action) and the confidence (max probability)
    confidences, predicted_classes_tensor = torch.max(probs_tensor, dim=1)
    
    predicted_classes = predicted_classes_tensor.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    all_probs_np = probs_tensor.cpu().numpy() # For raw_prediction

    predictions_list: List[Dict[str, Any]] = []
    for i in range(len(predicted_classes)):
        current_actual_price = actual_current_prices.iloc[i] if not actual_current_prices.empty and i < len(actual_current_prices) else None
        token_addr = token_addresses.iloc[i] if not token_addresses.empty and i < len(token_addresses) else None
        
        predicted_action_label = predicted_classes[i]
        # action_str = ACTION_MAP.get(predicted_action_label, "Unknown") # We'll add action string in API server based on Signal content
        confidence = confidences_np[i]
        class_probabilities = all_probs_np[i].tolist() # List of [P(Buy), P(Sell), P(Hold)]

        # For the API 'Signal' object, we need to provide what it expects.
        # 'predicted_next_price' is now set to current price.
        # 'raw_prediction' can store the class probabilities.
        # 'signal_strength' is the confidence.
        # An explicit 'action' field (Buy/Sell/Hold string) might be added to the Signal model in api_server.py
        # or derived there from raw_prediction.
        
        predictions_list.append({
            "token_address": token_addr,
            "current_price_actual": float(current_actual_price) if current_actual_price is not None else None,
            "predicted_next_price": float(current_actual_price) if current_actual_price is not None else None, # Set to current price
            "signal_strength": float(confidence),
            "raw_prediction": class_probabilities, # Store all class probabilities
            "predicted_action_label": int(predicted_action_label) # Keep the numeric label for potential direct use
        })
        
    return predictions_list

def get_predictions_for_api(
    tokens_filter: Optional[List[str]] = None,
    features_file_path: str = DEFAULT_FEATURES_PATH,
    model_path: str = MODEL_PATH,
    scaler_x_path: str = SCALER_X_PATH,
    # scaler_y_path: str = SCALER_Y_PATH, # REMOVED
    asset_encoder_path: str = ASSET_ENCODER_PATH,
    numerical_columns_list_path: str = NUMERICAL_COLUMNS_PATH
) -> List[Dict[str, Any]]:
    """High-level function for API to call, now for classification."""
    try:
        model, scaler_X, asset_encoder, trained_numerical_cols = load_model_and_scalers(
            model_path, scaler_x_path, asset_encoder_path, numerical_columns_list_path
        )
        
        try:
            all_features_df = pd.read_csv(features_file_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"Features file not found: {features_file_path}")

        if all_features_df.empty:
            print(f"Warning: Features file {features_file_path} is empty.")
            return []

        # For prediction, 'action_label' isn't strictly needed in the CSV, but 'price' (for current_price_actual) and features are.
        required_cols_for_prediction = trained_numerical_cols + ['token_address', 'price'] 
        
        missing_cols = [col for col in required_cols_for_prediction if col not in all_features_df.columns]
        if missing_cols:
            # Construct a more informative error message
            error_msg = (f"Features file {features_file_path} is missing required columns: {missing_cols}. "
                         f"Expected numerical columns based on training: {trained_numerical_cols}. "
                         f"Also need 'token_address' and 'price'.")
            # Add info about available columns if it's helpful for debugging
            # available_cols_str = ", ".join(all_features_df.columns.tolist())
            # error_msg += f" Available columns in CSV: {available_cols_str}" # Optional: can be very long
            raise ValueError(error_msg)

        # Filter by tokens if a filter is provided
        if tokens_filter and not all_features_df.empty:
            target_df = all_features_df[all_features_df['token_address'].isin(tokens_filter)].copy()
            if target_df.empty:
                print(f"Warning: No data found for specified tokens: {tokens_filter}")
                return []
        else:
            target_df = all_features_df.copy()

        # The crucial step: pass the `trained_numerical_cols` to generate_predictions.
        # `target_df` at this point must contain these columns (validated above).
        return generate_predictions(target_df, model, scaler_X, asset_encoder, trained_numerical_cols)

    except FileNotFoundError as e: 
        print(f"Error in get_predictions_for_api: {e}")
        raise
    except ValueError as e: # Catch ValueErrors from feature checks or missing columns
        print(f"ValueError in get_predictions_for_api: {e}")
        raise
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred in get_predictions_for_api: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(f"Internal error during prediction: {str(e)}")

if __name__ == '__main__':
    print("Testing prediction module...")
    
    if not os.path.exists("data"):
        os.makedirs("data")

    # Example: if your scaler_X expects 2 numerical features named 'feature1', 'feature2'
    # plus OHE features.
    # This dummy_numerical_feature_names list should reflect what you expect from training.
    dummy_numerical_feature_names = ['rsi', 'macd', 'cci', 'vwap_ratio'] # Example numerical features

    if not os.path.exists(NUMERICAL_COLUMNS_PATH): # Use the constant
        print(f"Warning: Dummy {NUMERICAL_COLUMNS_PATH} will be created for test.")
        with open(NUMERICAL_COLUMNS_PATH, 'w') as f:
            json.dump(dummy_numerical_feature_names, f)
        print(f"Created dummy {NUMERICAL_COLUMNS_PATH} with columns: {dummy_numerical_feature_names}")

    if not os.path.exists(DEFAULT_FEATURES_PATH):
        print(f"Warning: Dummy {DEFAULT_FEATURES_PATH} will be created for test.")
        # Dummy data should contain 'date', 'price', 'token_address', and all columns from dummy_numerical_feature_names
        # Also add 'action_label' for completeness, though it won't be used for inference input.
        dummy_data_dict = {
            'date': ['2023-01-01T00:00:00Z'], 
            'price': [100.0], 
            'token_address': ['DUMMY_TOKEN_ADDR'],
            'action_label': [2] # Example Hold label
        }
        for name in dummy_numerical_feature_names:
            dummy_data_dict[name] = [np.random.rand()] # One row of data
        
        pd.DataFrame(dummy_data_dict).to_csv(DEFAULT_FEATURES_PATH, index=False)
        print(f"Created dummy {DEFAULT_FEATURES_PATH} with columns: {list(dummy_data_dict.keys())}")

    # Dummy asset encoder - critical for standalone test to match OHE features logic
    if not os.path.exists(ASSET_ENCODER_PATH):
        print(f"Warning: Asset encoder at {ASSET_ENCODER_PATH} not found. Creating a dummy one.")
        # Fit on potential token addresses that might appear in DEFAULT_FEATURES_PATH
        # The dummy CSV has 'DUMMY_TOKEN_ADDR'. If your test uses other tokens via tokens_filter, add them here.
        dummy_encoder_df = pd.DataFrame({'token_address': ['DUMMY_TOKEN_ADDR', 'ANOTHER_TOKEN_ADDR']})
        # sparse_output=False is deprecated, use sparse=False for older sklearn, sparse_output for newer. Default is 'auto' or True.
        # For consistency with potential training setup, explicitly set handle_unknown.
        dummy_encoder = OneHotEncoder(handle_unknown='ignore') # sparse_output=False removed for wider compatibility
        dummy_encoder.fit(dummy_encoder_df[['token_address']])
        joblib.dump(dummy_encoder, ASSET_ENCODER_PATH)
        print(f"Dummy asset encoder saved to {ASSET_ENCODER_PATH}. Categories: {dummy_encoder.categories_}")

    # Dummy Scaler X - its n_features_in_ must match num_numerical + num_ohe
    if not os.path.exists(SCALER_X_PATH):
        print(f"Warning: Scaler X at {SCALER_X_PATH} not found. Creating a dummy one.")
        # n_features_in_ should be len(dummy_numerical_feature_names) + number of OHE features from dummy_encoder
        # For dummy_encoder with ['DUMMY_TOKEN_ADDR', 'ANOTHER_TOKEN_ADDR'], it produces 2 OHE features.
        num_ohe_dummy = sum(len(cat) for cat in joblib.load(ASSET_ENCODER_PATH).categories_)
        total_dummy_features = len(dummy_numerical_feature_names) + num_ohe_dummy
        
        dummy_scaler_X = StandardScaler()
        # Fit the scaler on some dummy data of the correct shape.
        # Create a dummy DataFrame that matches the structure scaler_X would expect:
        # Columns: dummy_numerical_feature_names + OHE feature names
        # For OHE names, we can use generic ones or derive from dummy_encoder if easy.
        # Let's use generic names for simplicity of fitting.
        # The actual names don't matter for fitting StandardScaler if data is numpy.
        # What matters is n_features_in_ gets set.
        dummy_scaler_X.fit(np.random.rand(2, total_dummy_features)) # Fit on 2 rows, total_dummy_features columns
        joblib.dump(dummy_scaler_X, SCALER_X_PATH)
        print(f"Dummy Scaler X saved to {SCALER_X_PATH}. Expected features (n_features_in_): {dummy_scaler_X.n_features_in_}")
        if dummy_scaler_X.n_features_in_ != total_dummy_features:
             print(f"CRITICAL WARNING: Dummy Scaler X n_features_in_ ({dummy_scaler_X.n_features_in_}) "
                   f"does not match calculated total_dummy_features ({total_dummy_features}). Test may fail.")


    # Dummy Model File - its input_dim must match Scaler X's n_features_in_, output_dim=3
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model at {MODEL_PATH} not found. Creating a dummy one.")
        # Ensure input_dim matches the dummy_scaler_X.n_features_in_
        try:
            scaler_x_for_model_dim = joblib.load(SCALER_X_PATH)
            model_input_dim = scaler_x_for_model_dim.n_features_in_
            dummy_model = Net(input_dim=model_input_dim, output_dim=3) # Ensure output_dim=3 for classification dummy model
            # Save an empty state_dict, or one with random weights if Net initializes them
            torch.save(dummy_model.state_dict(), MODEL_PATH)
            print(f"Dummy model saved to {MODEL_PATH} with input_dim={model_input_dim}")
        except Exception as model_ex:
            print(f"Error creating dummy model: {model_ex}. Standalone test might fail severely.")


    try:
        print("\\n--- Testing with no token filter (using dummy/actual artifacts) ---")
        # This call will now use numerical_columns_list_path = NUMERICAL_COLUMNS_PATH by default
        sample_predictions = get_predictions_for_api() 
        
        print(f"Generated {len(sample_predictions)} predictions.")
        if sample_predictions:
            print("First prediction:", sample_predictions[0])
        else:
            print("No predictions generated or an empty list was returned.")
    
    except FileNotFoundError as e:
        print(f"Test failed: Missing file - {e}")
    except ValueError as e:
        print(f"Test failed: Value error - {e}")
    except RuntimeError as e:
        print(f"Test failed: Runtime error - {e}")
    except Exception as e:
        import traceback
        print(f"Error during direct test of prediction.py: {str(e)}")
        traceback.print_exc()

    # Example of how to call with a specific token
    # try:
    #     print("\\nAttempting to get predictions for 'DUMMY_TOKEN_ADDR'...")
    #     # This token is in the dummy CSV and dummy asset encoder
    #     specific_token_predictions = get_predictions_for_api(tokens_filter=['DUMMY_TOKEN_ADDR'])
    #     if specific_token_predictions:
    #         print(f"Successfully got {len(specific_token_predictions)} prediction(s) for 'DUMMY_TOKEN_ADDR'. First one: {specific_token_predictions[0]}")
    #     else:
    #         print(f"Got no predictions for 'DUMMY_TOKEN_ADDR'.")
    # except Exception as e:
    #     print(f"Error during specific token test: {e}")


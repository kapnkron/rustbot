import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split # For splitting train/val
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import joblib
import json # Ensure json is imported
import argparse # Add argparse import

# Import the network definition
from .your_model_definition_file import Net # Use . for relative import within package

# Configure logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')

# Default paths for artifacts (can be overridden)
DEFAULT_MODEL_DIR = 'models/neural_network'
DEFAULT_TRAIN_FEATURES_PATH = 'data/features_train_ohlcv.csv' # This will now contain 'action_label'

def run_training_pipeline(
    data_path: str = DEFAULT_TRAIN_FEATURES_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    val_size: float = 0.2, # Proportion of data to use for validation
    target_column: str = 'action_label', # CHANGED: Target is now 'action_label'
    # Model hyperparameters
    hidden_dim1: int = 64,
    hidden_dim2: int = 32,
    dropout_rate: float = 0.2,
    # Training hyperparameters
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 20,
    # Scaler choice for X is still relevant, y is not scaled for classification
    # use_robust_scaler_y: bool = True, # REMOVED: y is not scaled for classification
    random_state: Optional[int] = 42
) -> Dict:
    """
    Loads data, trains a neural network model (Net) for classification, and saves the model and scalers.

    Returns:
        A dictionary containing training status, paths to artifacts, and metrics.
    """
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Starting training pipeline run: {run_timestamp}")
    
    # Create a specific subdirectory for this run's artifacts
    run_specific_model_dir = os.path.join(model_dir, run_timestamp)
    os.makedirs(run_specific_model_dir, exist_ok=True)
    logger.info(f"Artifacts for this run will be saved in: {run_specific_model_dir}")
    
    results = {
        "status": "failed",
        "message": "",
        "run_timestamp": run_timestamp,
        "artifacts_dir": run_specific_model_dir, # Store the specific dir
        "model_path": None,
        "scaler_X_path": None,
        "scaler_y_path": None,
        "asset_encoder_path": None,
        "numerical_columns_path": None, # Added for the new artifact
        "training_log_path": None, 
        "final_train_loss": None,
        "final_val_loss": None,
        "epochs_trained": None,
        "input_data_path": data_path,
        "model_architecture": "Net"
    }

    try:
        # Model directory is now the run_specific_model_dir
        logger.info(f"Using model directory: {run_specific_model_dir}")

        # Load data
        if not os.path.exists(data_path):
            results["message"] = f"Data file not found: {data_path}"
            logger.error(results["message"])
            return results
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Handle potential NaNs from feature engineering (especially leading NaNs)
        df.dropna(subset=[target_column], inplace=True) # Target must not be NaN
        # For features, we'll use np.nan_to_num later, but let's see initial drop impact
        # df.dropna(inplace=True) # This was too aggressive before. Revisit if needed.
        logger.info(f"Shape after dropping rows with NaN target ('{target_column}'): {df.shape}")

        if df.empty:
            results["message"] = f"DataFrame is empty after dropping NaNs in target '{target_column}'."
            logger.error(results["message"])
            return results

        # Define features (X) and target (y)
        # Exclude target and any identifier columns from features
        # 'token_address' is used for OHE, other non-numeric might need to be dropped or encoded
        # For now, assume 'token_address' is the main categorical feature to one-hot encode
        
        categorical_features = ['token_address']
        # This is the key list of numerical features we need to save
        numerical_feature_names = [col for col in df.columns if col not in [target_column, 'date'] + categorical_features and df[col].dtype in [np.float64, np.int64]]

        if not numerical_feature_names: # Changed variable name here for clarity
            results["message"] = "No numerical features found after excluding target, date, and token_address."
            logger.error(results["message"])
            return results
            
        logger.info(f"Numerical features ({len(numerical_feature_names)}): {numerical_feature_names}")
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

        # One-Hot Encode 'token_address'
        # This asset_encoder should be the same one used for prediction.
        # We need to fit it on the training data and save it.
        # In prediction, it must be loaded.
        from sklearn.preprocessing import OneHotEncoder
        asset_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit and transform on the full dataset before splitting to ensure all tokens are known
        encoded_assets = asset_encoder.fit_transform(df[categorical_features])
        encoded_asset_cols = asset_encoder.get_feature_names_out(categorical_features)
        encoded_assets_df = pd.DataFrame(encoded_assets, columns=encoded_asset_cols, index=df.index)
        
        X_numerical = df[numerical_feature_names].copy() # Use the saved list
        X = pd.concat([X_numerical, encoded_assets_df], axis=1)
        
        # Target y is now the 'action_label' column, and should be long integers for NLLLoss
        y = df[target_column].values # No reshape, NLLLoss expects 1D tensor of class indices
        
        logger.info(f"Shape of X (features) before scaling: {X.shape}")
        logger.info(f"Shape of y (target classes '{target_column}'): {y.shape}")

        # Split data into training and validation sets
        # Ensure y is treated as discrete labels for stratification if desired, though not strictly needed for NLLLoss if classes are imbalanced
        X_train_df, X_val_df, y_train_labels, y_val_labels = train_test_split(
            X, y, test_size=val_size, random_state=random_state, shuffle=False, # Time series data, usually no shuffle
            # stratify=y if np.unique(y).size > 1 and not any(np.isnan(y)) else None # Optional: Stratify if labels are clean and multi-class
        )
        logger.info(f"Training set shape: X-{X_train_df.shape}, y-{y_train_labels.shape}")
        logger.info(f"Validation set shape: X-{X_val_df.shape}, y-{y_val_labels.shape}")

        # Impute NaNs in features (e.g., from rolling calculations at the start) with 0
        # This should be done after splitting to prevent data leakage from val to train if using means/medians
        X_train = np.nan_to_num(X_train_df.values, nan=0.0)
        X_val = np.nan_to_num(X_val_df.values, nan=0.0)

        # Scale features (X)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        logger.info("Features (X) scaled using StandardScaler.")
        
        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        # y_train_tensor should be LongTensor for NLLLoss
        y_train_tensor = torch.LongTensor(y_train_labels).to(device) 
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        # y_val_tensor should be LongTensor for NLLLoss
        y_val_tensor = torch.LongTensor(y_val_labels).to(device)
        
        logger.info(f"Final shapes for training - X_train: {X_train_tensor.shape}, y_train ('{target_column}'): {y_train_tensor.shape}")
        logger.info(f"Final shapes for validation - X_val: {X_val_tensor.shape}, y_val ('{target_column}'): {y_val_tensor.shape}")

        # Initialize model using Net definition for classification (output_dim=3)
        input_dim = X_train_tensor.shape[1]
        model = Net(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=3, dropout_rate=dropout_rate).to(device)
        logger.info(f"Model initialized for classification: {model}")

        criterion = nn.NLLLoss() # CHANGED: Use NLLLoss with LogSoftmax output
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for i in range(0, len(X_train_tensor), batch_size):
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), batch_size):
                    X_val_batch = X_val_tensor[i:i+batch_size]
                    y_val_batch = y_val_tensor[i:i+batch_size]
                    val_outputs = model(X_val_batch)
                    val_loss = criterion(val_outputs, y_val_batch)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model state
                best_model_state = model.state_dict()
                results["final_train_loss"] = avg_train_loss
                results["final_val_loss"] = avg_val_loss
                results["epochs_trained"] = epoch + 1
                logger.info(f"Validation loss improved to {avg_val_loss:.6f}. Saving model state for this epoch.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement. Best Val Loss: {best_val_loss:.6f}")
                break
        
        if best_model_state is None: # Changed from 'best_model_state' not in locals()
            logger.warning("Training completed without any improvement in validation loss or did not run long enough. Saving last model state.")
            best_model_state = model.state_dict() # Save the last state if no improvement
            if results["epochs_trained"] is None: 
                 results["epochs_trained"] = epochs
                 results["final_train_loss"] = avg_train_loss # Log last train/val loss
                 results["final_val_loss"] = avg_val_loss


        # Save the best model and scalers
        # Filenames no longer include timestamp as they are in a timestamped directory
        model_filename = "model_net.pth"
        scaler_X_filename = "scaler_X.pkl"
        asset_encoder_filename = "asset_encoder.pkl"
        numerical_columns_filename = "numerical_columns.json" # New artifact filename

        # Paths are now within run_specific_model_dir
        model_path = os.path.join(run_specific_model_dir, model_filename)
        scaler_X_path = os.path.join(run_specific_model_dir, scaler_X_filename)
        asset_encoder_path = os.path.join(run_specific_model_dir, asset_encoder_filename)
        numerical_columns_path = os.path.join(run_specific_model_dir, numerical_columns_filename) # Path for the new artifact

        if best_model_state: # Ensure there's a model state to save
            torch.save(best_model_state, model_path)
            results["model_path"] = model_path
            logger.info(f"Best model saved to {model_path}")
        else:
            results["message"] += " No best model state found to save."
            logger.error("Critical: No best_model_state was captured to save.")


        joblib.dump(scaler_X, scaler_X_path)
        results["scaler_X_path"] = scaler_X_path
        logger.info(f"Scaler_X saved to {scaler_X_path}")
        
        joblib.dump(asset_encoder, asset_encoder_path)
        results["asset_encoder_path"] = asset_encoder_path
        logger.info(f"Asset encoder saved to {asset_encoder_path}")

        # Save the numerical feature names
        with open(numerical_columns_path, 'w') as f:
            json.dump(numerical_feature_names, f) # Use numerical_feature_names identified earlier
        results["numerical_columns_path"] = numerical_columns_path
        logger.info(f"Numerical column names saved to {numerical_columns_path}")
        
        results["status"] = "success"
        results["message"] = f"Training completed. Model and artifacts saved in {run_specific_model_dir}. Symlink 'latest' points to this run."

        # Clean up results dictionary for printing: remove large data like feature names
        # Create a copy for logging to avoid modifying the original dict if it's used later
        results_for_log = results.copy()
        if 'numerical_feature_names' in results_for_log: # Check if key exists
            del results_for_log['numerical_feature_names']

        logger.info(f"Training pipeline finished with results: {results_for_log}") # Log the cleaned-up results

        # --- Symlink automation for latest model ---
        try:
            latest_symlink = os.path.join(model_dir, "latest") # e.g., models/neural_network/latest
            # run_specific_model_dir is e.g., models/neural_network/20250521_235426

            # Get just the timestamped folder name for the symlink source
            source_folder_name = os.path.basename(run_specific_model_dir) # e.g., 20250521_235426

            # Remove existing symlink if it exists
            if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
                os.remove(latest_symlink)
            
            # Symlink source should be relative to the link's directory (model_dir)
            # The target for os.symlink should be the path that the link points to.
            # If latest_symlink is models/neural_network/latest,
            # and source_folder_name is 20250521_235426,
            # this creates `models/neural_network/latest` -> `20250521_235426` (relative to models/neural_network/)
            os.symlink(source_folder_name, latest_symlink)
            logger.info(f"Successfully created/updated symlink: {latest_symlink} -> {source_folder_name}") # Use source_folder_name
        except Exception as symlink_err:
            logger.error(f"Failed to create/update symlink for latest model: {symlink_err}")

    except FileNotFoundError as fnf_error:
        results["message"] = f"File not found error: {str(fnf_error)}"
        logger.error(results["message"])
    except ValueError as val_error:
        results["message"] = f"Value error during training: {str(val_error)}"
        logger.error(results["message"], exc_info=True)
    except RuntimeError as rt_error: # Catch PyTorch runtime errors
        results["message"] = f"PyTorch runtime error during training: {str(rt_error)}"
        logger.error(results["message"], exc_info=True)
    except Exception as e:
        results["message"] = f"An unexpected error occurred during training: {str(e)}"
        logger.error(results["message"], exc_info=True)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML training pipeline.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_FEATURES_PATH,
                        help=f"Path to the training data CSV file. Default: {DEFAULT_TRAIN_FEATURES_PATH}")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save model artifacts. Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion of data to use for validation. Default: 0.2")
    parser.add_argument("--target_column", type=str, default='action_label',
                        help="Name of the target column in the data. Default: 'action_label'")
    parser.add_argument("--hidden_dim1", type=int, default=64,
                        help="Number of units in the first hidden layer. Default: 64")
    parser.add_argument("--hidden_dim2", type=int, default=32,
                        help="Number of units in the second hidden layer. Default: 32")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                        help="Dropout rate for dropout layers. Default: 0.2")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs. Default: 200")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training. Default: 32")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer. Default: 0.001")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience for early stopping. Default: 20")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility. Default: 42")
    
    args = parser.parse_args()

    logger.info(f"Running training with arguments: {args}")
    
    run_training_pipeline(
        data_path=args.data_path,
        model_dir=args.model_dir,
        val_size=args.val_size,
        target_column=args.target_column,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        random_state=args.random_state
    )

    if args.model_dir == DEFAULT_MODEL_DIR:
        logger.info("To use these artifacts for prediction, copy them from:")
        logger.info(f"  {args.model_dir}/latest")
        logger.info("to the root project directory (or where your prediction script expects them), renaming them to:")
        logger.info(f"  model_trading.pt (from {os.path.basename(args.model_dir + '/latest')})")
        logger.info(f"  scaler_X_trading.pkl (from {os.path.basename(args.model_dir + '/latest')})")
        logger.info(f"  asset_encoder_trading.pkl (from {os.path.basename(args.model_dir + '/latest')})")
        logger.info(f"  numerical_columns.json (from {os.path.basename(args.model_dir + '/latest')})") 
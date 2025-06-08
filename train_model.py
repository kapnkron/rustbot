import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
import glob
import numpy as np
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train price prediction models')
parser.add_argument('--unified', action='store_true', help='Train a single unified model for all tokens')
parser.add_argument('--separate', action='store_true', help='Train separate models for general tokens and pump graduates')
args = parser.parse_args()

# Default to training both models if no specific flag is provided
if not args.unified and not args.separate:
    args.unified = True
    args.separate = True

# Create a timestamp for model versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = Path(f"models/neural_network/{timestamp}")
models_dir.mkdir(parents=True, exist_ok=True)

# Create a symlink to the latest model directory
latest_dir = Path("models/neural_network/latest")
if latest_dir.exists() or latest_dir.is_symlink():
    try:
        latest_dir.unlink()
    except:
        logging.warning(f"Could not unlink {latest_dir}, trying to continue...")
latest_dir.symlink_to(models_dir.relative_to(latest_dir.parent.parent))

print(f"Models will be saved to: {models_dir}")

# Try different sources of data
try:
    # 1. Check for features_train_ohlcv.csv
    features_path = 'data/features_train_ohlcv.csv'
    if os.path.exists(features_path) and os.path.getsize(features_path) > 0:
        logging.info(f"Using features from {features_path}")
        merged = pd.read_csv(features_path)
    else:
        # 2. Check for legacy features_ohlcv.csv
        legacy_path = 'data/features_ohlcv.csv'
        if os.path.exists(legacy_path) and os.path.getsize(legacy_path) > 0:
            logging.info(f"Using legacy features from {legacy_path}")
            merged = pd.read_csv(legacy_path)
        else:
            # 3. Try to find raw OHLCV data
            ohlcv_files = glob.glob("data/processed/DATA_*.csv")
            if ohlcv_files:
                logging.info(f"Found {len(ohlcv_files)} raw OHLCV files, combining them")
                dfs = []
                for f in ohlcv_files:
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        logging.warning(f"Error reading {f}: {e}")
                
                if dfs:
                    merged = pd.concat(dfs, ignore_index=True)
                    logging.info(f"Combined {len(dfs)} files into a dataset with shape {merged.shape}")
                else:
                    raise ValueError("No valid data found in OHLCV files")
            else:
                # No data found
                logging.error("No training data found. Please run data collection and feature engineering first.")
                sys.exit(0)  # Exit gracefully
    
    # Check if we have enough data
    if len(merged) < 100:
        logging.warning(f"Only {len(merged)} data points available - need at least 100 for meaningful training")
        logging.info("Waiting for more data to be collected...")
        sys.exit(0)  # Exit gracefully

    # 1. Read the pool_addresses.json to get token categories
    try:
        with open('pool_addresses.json', 'r') as f:
            pool_addresses = json.load(f)
    except FileNotFoundError:
        logging.warning("Warning: pool_addresses.json not found. All tokens will be treated as general tokens.")
        pool_addresses = {}

    # Build a mapping of token address to category
    token_categories = {}
    for market_name, info in pool_addresses.items():
        base_address = info.get('base', {}).get('address')
        if base_address:
            category = info.get('category', 'general_token')
            token_categories[base_address] = category

    # Add category column to the dataset if 'token_address' exists
    if 'token_address' in merged.columns:
        merged['category'] = merged['token_address'].map(token_categories).fillna('general_token')
        print(f"Token categories: {merged['category'].value_counts()}")
    else:
        # Create a default category
        merged['category'] = 'general_token'
        logging.warning("No token_address column found, using 'general_token' for all data")
        if 'token' in merged.columns:
            logging.info(f"Found 'token' column with values: {merged['token'].unique()}")
        
    # Ensure there's a token_address column even if it's not in the original data
    if 'token_address' not in merged.columns:
        if 'token' in merged.columns:
            merged['token_address'] = merged['token']
        else:
            merged['token_address'] = 'unknown'
            logging.warning("No token identification column found, using 'unknown' for all data")

    # 3. Prepare features and targets
    # Exclude non-feature columns
    excluded_cols = ['date', 'price', 'token_address', 'category', 'signature', 'timestamp', 'side']
    features = [col for col in merged.columns if col not in excluded_cols]
    
    # Check if 'price' column exists, otherwise use 'close'
    if 'price' not in merged.columns and 'close' in merged.columns:
        logging.info("Using 'close' column as price")
        merged['price'] = merged['close']
    elif 'price' not in merged.columns:
        logging.error("No 'price' or 'close' column found in the data")
        sys.exit(1)
        
    X_raw = merged[features]
    asset_col = merged['token_address'].values.reshape(-1, 1)

    # Print info about merged['price'] before shift
    print(f"Shape of merged['price']: {merged['price'].shape}")
    print(f"Sample of merged['price']:\n{merged['price'].head()}")
    print(f"NaNs in merged['price']: {merged['price'].isnull().sum()}")

    # Target: next-period price (regression)
    y_raw = merged['price'].shift(-1)

    # Print info about y_raw after shift
    print(f"Shape of y_raw after shift: {y_raw.shape}")
    print(f"Sample of y_raw after shift:\n{y_raw.head()}")
    print(f"NaNs in y_raw after shift: {y_raw.isnull().sum()}")

    # Align X and y by dropping NaNs from y
    merged_filtered = merged.loc[y_raw.dropna().index]
    if merged_filtered.empty:
        raise ValueError("No samples remaining after aligning X and y (merged_filtered is empty).")

    X_filtered = merged_filtered[features]
    y_filtered = y_raw.loc[merged_filtered.index]
    categories_filtered = merged_filtered['category']

    # Print info after initial NaN drop based on y_raw
    print(f"Shape of X_filtered after initial y_raw.dropna(): {X_filtered.shape}")
    print(f"Shape of y_filtered after initial y_raw.dropna(): {y_filtered.shape}")
    print(f"NaNs in y_filtered: {y_filtered.isnull().sum()}")

    if X_filtered.empty or y_filtered.empty:
        raise ValueError(f"X_filtered or y_filtered is empty before preprocessing.")

    # One-hot encode asset
    asset_col_filtered = merged_filtered['token_address'].values.reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    asset_encoded = enc.fit_transform(asset_col_filtered)

    # Combine numerical features and encoded asset features
    X = np.hstack([X_filtered.values, asset_encoded])
    y = y_filtered.values

    # Print shapes before final check
    print(f"Shape of X before final check: {X.shape}")
    print(f"Shape of y before final check: {y.shape}")

    # Apply log transform to y before scaling
    print("\n--- Applying log1p transform to y ---")
    y = np.maximum(y, 0)  # Ensure no negative prices before log
    y_log = np.log1p(y)

    # Check for NaNs/Infs in X before scaling
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Use RobustScaler for y_log
    scaler_y = RobustScaler()
    y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1))

    # Save the preprocessing objects
    joblib_dir = models_dir / "joblib"
    joblib_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(scaler_X, joblib_dir / 'scaler_X.pkl')
    joblib.dump(scaler_y, joblib_dir / 'scaler_y.pkl')
    joblib.dump(enc, joblib_dir / 'asset_encoder.pkl')

except Exception as e:
    logging.error(f"Error during data preparation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Define an enhanced neural network model
class EnhancedNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, output_dim=1):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Apply weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

# 5. Function to train a model
def train_model(X_train, y_train, X_val, y_val, input_dim, model_name, hidden_dims=[128, 64, 32], dropout_rate=0.3):
    model = EnhancedNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    PATIENCE = 100
    MIN_DELTA = 1e-5
    MAX_EPOCHS = 10000
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # For plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss.item() < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save the best model
    model_path = models_dir / f"{model_name}.pt"
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
    else:
        print("Warning: No improvement in validation loss. Saving current model state.")
        torch.save(model.state_dict(), model_path)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    plt.savefig(models_dir / f"{model_name}_loss_curve.png")
    
    return best_val_loss

# 6. Prepare data for training
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Train a unified model if requested
if args.unified:
    print("\n=== Training Unified Model ===")
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    best_val_loss = train_model(X_train, y_train, X_test, y_test, X_train.shape[1], "unified_model")
    print(f"Unified model trained. Best validation loss: {best_val_loss:.6f}")

# Train separate models for general tokens and pump graduates if requested
if args.separate:
    # Get indices for each category
    general_indices = [i for i, cat in enumerate(categories_filtered) if cat == 'general_token']
    pump_indices = [i for i, cat in enumerate(categories_filtered) if cat == 'pump_graduate']
    
    if pump_indices:  # Only train if we have pump graduate tokens
        print("\n=== Training Pump Graduate Model ===")
        X_pump = X_tensor[pump_indices]
        y_pump = y_tensor[pump_indices]
        
        X_train_pump, X_test_pump, y_train_pump, y_test_pump = train_test_split(X_pump, y_pump, test_size=0.2, random_state=42)
        pump_val_loss = train_model(X_train_pump, y_train_pump, X_test_pump, y_test_pump, X_train_pump.shape[1], "pump_graduate_model")
        print(f"Pump graduate model trained. Best validation loss: {pump_val_loss:.6f}")
    else:
        print("No 'pump_graduate' tokens found in the dataset. Skipping pump graduate model training.")
    
    if general_indices:  # Only train if we have general tokens
        print("\n=== Training General Token Model ===")
        X_general = X_tensor[general_indices]
        y_general = y_tensor[general_indices]
        
        X_train_general, X_test_general, y_train_general, y_test_general = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
        general_val_loss = train_model(X_train_general, y_train_general, X_test_general, y_test_general, X_train_general.shape[1], "general_token_model")
        print(f"General token model trained. Best validation loss: {general_val_loss:.6f}")
    else:
        print("No 'general_token' tokens found in the dataset. Skipping general token model training.")

print(f"\nTraining complete. Models saved to {models_dir}")
print(f"Latest model symlink created at {latest_dir}") 
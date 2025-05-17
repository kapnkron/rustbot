import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
import glob
import numpy as np

# 1. Read all three top token lists
lists = [
    'top_100_solana_gainers.csv',
    'top_100_solana_losers.csv',
    'top_100_solana_hotpools.csv'
]
token_addresses = set()
for fname in lists:
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    # Try to extract mainToken and sideToken addresses
    for col in ['mainToken', 'sideToken']:
        if col in df.columns:
            token_addresses.update(df[col].dropna().apply(lambda x: eval(x)['address'] if isinstance(x, str) and x.startswith('{') else x))

# 2. Merge feature CSVs for these tokens
# Use the merged features file directly
features_path = 'data/features_train_ohlcv.csv'
if not os.path.exists(features_path):
    raise RuntimeError(f'Features file not found: {features_path}. Please run scripts/add_features.py first.')
merged = pd.read_csv(features_path)

# 3. Prepare features and targets
features = [col for col in merged.columns if col not in ['date', 'price', 'token_address']]
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
    raise ValueError("No samples remaining after aligning X and y (merged_filtered is empty). This might be due to y_raw being all NaNs or X_raw having NaNs that cause all rows to be dropped.")

X_filtered = merged_filtered[features]
y_filtered = y_raw.loc[merged_filtered.index] # Ensure y is filtered using the same index

# Print info after initial NaN drop based on y_raw
print(f"Shape of X_filtered after initial y_raw.dropna(): {X_filtered.shape}")
print(f"Shape of y_filtered after initial y_raw.dropna(): {y_filtered.shape}")
print(f"NaNs in y_filtered: {y_filtered.isnull().sum()}")


if X_filtered.empty or y_filtered.empty:
    raise ValueError(f"X_filtered (shape {X_filtered.shape}) or y_filtered (shape {y_filtered.shape}) is empty before OneHotEncoding. Check data and NaN handling.")

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

# ---> Apply log transform to y before scaling
print("\n--- Applying log1p transform to y ---")
# Ensure y is non-negative for log1p. Clip very small negative values if any, though prices should be >= 0.
y = np.maximum(y, 0) # Ensure no negative prices before log
y_log = np.log1p(y)
print(f"y_log min: {np.min(y_log):.8f}, y_log max: {np.max(y_log):.8f}")
print(f"y_log mean: {np.mean(y_log):.8f}, y_log std: {np.std(y_log):.8f}")

if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError(f"No samples to train on after preprocessing. X shape: {X.shape}, y shape: {y.shape}")
if X.shape[0] != y.shape[0]:
    raise ValueError(f"X and y have mismatched sample numbers: X ({X.shape[0]}), y ({y.shape[0]})")

print(f"Final shapes - X: {X.shape}, y: {y.shape}")

# Check for NaNs/Infs in X before scaling
print(f"NaNs in X before scaling: {np.isnan(X).sum()}")
print(f"Infs in X before scaling: {np.isinf(X).sum()}")

# Impute NaNs in X with 0 before scaling
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Also handle potential infs
print(f"NaNs in X after imputation: {np.isnan(X).sum()}")
print(f"Infs in X after imputation: {np.isinf(X).sum()}")

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# ---> Modify y stats printing for y_log
print("\n--- Target Variable (y_log) Stats Before Scaling ---")
print(f"y_log shape: {y_log.shape}")
print(f"y_log min: {np.min(y_log):.8f}, y_log max: {np.max(y_log):.8f}")
print(f"y_log mean: {np.mean(y_log):.8f}, y_log std: {np.std(y_log):.8f}")
print(f"Number of NaNs in y_log: {np.isnan(y_log).sum()}, Infs: {np.isinf(y_log).sum()}")

# ---> CHANGE: Use RobustScaler for y_log instead of StandardScaler
# scaler_y = StandardScaler()
scaler_y = RobustScaler() # Use RobustScaler
y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)) # Scale the log-transformed y

# ---> Add prints for scaler_y attributes and y_scaled stats
print("\n--- RobustScaler for y (scaler_y) --- ") # Updated comment
# RobustScaler has center_ (median) and scale_ (IQR)
if hasattr(scaler_y, 'center_') and scaler_y.center_ is not None:
    print(f"scaler_y.center_ (median): {scaler_y.center_[0]:.8f}")
if hasattr(scaler_y, 'scale_') and scaler_y.scale_ is not None:
    print(f"scaler_y.scale_ (IQR): {scaler_y.scale_[0]:.8f}")

print("\n--- Target Variable (y_log) Detailed Stats Before Scaling ---")
print(f"y_log | Min: {np.min(y_log):.8f}, Max: {np.max(y_log):.8f}, Mean: {np.mean(y_log):.8f}, Std: {np.std(y_log):.8f}")
percentiles_log = [1, 5, 25, 50, 75, 95, 99]
print(f"y_log Percentiles: { {p: np.percentile(y_log, p) for p in percentiles_log} }")

print("\n--- Target Variable (y_scaled) Stats After Scaling ---")
print(f"y_scaled shape: {y_scaled.shape}")
# y_scaled is 2D, so access elements appropriately for min/max/mean/std
print(f"y_scaled | Min: {np.min(y_scaled):.8f}, Max: {np.max(y_scaled):.8f}, Mean: {np.mean(y_scaled):.8f}, Std: {np.std(y_scaled):.8f}")
percentiles_scaled = [1, 5, 25, 50, 75, 95, 99]
print(f"y_scaled Percentiles: { {p: np.percentile(y_scaled, p) for p in percentiles_scaled} }")
print(f"Number of NaNs in y_scaled: {np.isnan(y_scaled).sum()}, Infs: {np.isinf(y_scaled).sum()}")


# Check for NaNs/Infs after scaling
print(f"NaNs in X_scaled: {np.isnan(X_scaled).sum()}")
print(f"Infs in X_scaled: {np.isinf(X_scaled).sum()}")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 4. Define and train the model
class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

output_dim = 1
criterion = nn.MSELoss()
model = Net(input_dim=X_train.shape[1], output_dim=output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

PATIENCE = 100
MIN_DELTA = 1e-4
MAX_EPOCHS = 10000
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(MAX_EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    model.eval()
    val_outputs = model(X_test.to(device))
    val_loss = criterion(val_outputs, y_test.to(device))
    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    if val_loss.item() < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

model.cpu()
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, 'model_trading.pt')
else:
    print("Warning: No improvement in validation loss. Saving current model state.")
    torch.save(model.state_dict(), 'model_trading.pt')

import joblib
joblib.dump(scaler_X, 'scaler_X_trading.pkl')
joblib.dump(scaler_y, 'scaler_y_trading.pkl')
joblib.dump(enc, 'asset_encoder_trading.pkl')
print("Unified model and scalers saved as model_trading.pt, scaler_X_trading.pkl, scaler_y_trading.pkl, asset_encoder_trading.pkl") 
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import numpy as np

# Define Net at the top level so it can be pickled
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

# Use the feature-rich CSVs
features_dir = 'data/features'
feature_files = glob.glob(os.path.join(features_dir, '*_features.csv'))

# Targets to train on
TARGETS = {
    'close': lambda df: df['close'].shift(-1),
    'return': lambda df: (df['close'].shift(-1) / df['close']) - 1,
    'direction': lambda df: (df['close'].shift(-1) > df['close']).astype(int),
    'big_move': lambda df: (np.abs((df['close'].shift(-1) / df['close']) - 1) > 0.02).astype(int),
}

# Early stopping params
PATIENCE = 100
MIN_DELTA = 1e-4
MAX_EPOCHS = 10000

for feature_file in feature_files:
    asset = os.path.basename(feature_file).split('_')[0].upper()
    print(f"\n=== Training models for {asset} ===")
    df = pd.read_csv(feature_file)
    features = [col for col in df.columns if col not in ['date', 'close']]
    for target_name, target_func in TARGETS.items():
        print(f"\n--- Target: {target_name} ---")
        # Prepare X and y
        X = df[features].values[:-1]
        y_raw = target_func(df)[:-1].values.reshape(-1, 1)
        # Remove NaNs (from shifting)
        mask = ~np.isnan(y_raw).flatten()
        X = X[mask]
        y_raw = y_raw[mask]
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        if target_name in ['close', 'return']:
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_raw)
        else:
            scaler_y = None
            y_scaled = y_raw  # classification, no scaling
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
        if target_name in ['direction', 'big_move']:
            output_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        else:
            output_dim = 1
            criterion = nn.MSELoss()
        model = Net(input_dim=X_train.shape[1], output_dim=output_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.to(device))
            if target_name in ['direction', 'big_move']:
                loss = criterion(outputs, y_train.to(device))
            else:
                loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()
            # Always check validation loss and print every epoch
            model.eval()
            val_outputs = model(X_test.to(device))
            if target_name in ['direction', 'big_move']:
                val_loss = criterion(val_outputs, y_test.to(device))
            else:
                val_loss = criterion(val_outputs, y_test.to(device))
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            # Early stopping
            if val_loss.item() < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        # Save best model and scalers
        model.cpu()
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        torch.save(model, f'model_{asset}_{target_name}.pt')
        import joblib
        joblib.dump(scaler_X, f'scaler_X_{asset}_{target_name}.pkl')
        if scaler_y is not None:
            joblib.dump(scaler_y, f'scaler_y_{asset}_{target_name}.pkl')
        print(f"Model and scalers saved as model_{asset}_{target_name}.pt, scaler_X_{asset}_{target_name}.pkl" + (f", scaler_y_{asset}_{target_name}.pkl" if scaler_y is not None else "")) 
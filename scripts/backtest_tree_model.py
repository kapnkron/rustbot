import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import os

input_path = 'data/features_ohlcv.csv'
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Feature file not found: {input_path}")

df = pd.read_csv(input_path, parse_dates=['date'])

# Predict next-day return direction (1 if return > 0, else 0)
# Ensure 'return' and 'token_address' columns exist
if 'return' not in df.columns:
    raise ValueError("'return' column not found. Please ensure it is present in features_ohlcv.csv")
if 'token_address' not in df.columns:
    raise ValueError("'token_address' column not found. Please ensure it is present in features_ohlcv.csv")

df['target'] = (df.groupby('token_address')['return'].shift(-1) > 0).astype(int)

print(f"Shape of df BEFORE any NaN handling: {df.shape}")
print(f"NaNs in target column immediately after creation: {df['target'].isnull().sum()}")

# Identify and drop columns that are entirely NaN
all_nan_cols = df.columns[df.isnull().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN columns: {all_nan_cols}")
    df = df.drop(columns=all_nan_cols)
    print(f"Shape of df after dropping all-NaN columns: {df.shape}")

# Define feature columns from the *remaining* columns
# These are the columns we intend to use as features for the model
feature_cols = [col for col in df.columns if col not in ['date', 'token_address', 'target'] and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
print(f"Selected feature_cols for model training and further NaN checks: {feature_cols}")

# Drop rows where 'target' is NaN, or where any of the selected feature_cols are NaN.
print(f"Shape of df BEFORE final dropna based on target and selected features: {df.shape}")
print(f"NaNs in target before final dropna: {df['target'].isnull().sum()}")
# For detailed diagnostics, let's check NaNs in selected feature_cols before dropping
# for col in feature_cols:
#     if df[col].isnull().any():
#         print(f"  NaNs in feature column '{col}' before final dropna: {df[col].isnull().sum()}")

df = df.dropna(subset=['target'] + feature_cols)
print(f"Shape of df AFTER final dropna: {df.shape}")

df = df.reset_index(drop=True)

if df.empty:
    print("DataFrame is empty after all NaN handling. Exiting to avoid TimeSeriesSplit error.")
    exit()

# The feature_cols list is already defined correctly above for X
print(f"Final feature columns used for X: {feature_cols}")
X = df[feature_cols].values
y = df['target'].values

# TimeSeriesSplit for walk-forward validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
accuracies = []
feature_importances = np.zeros(len(feature_cols))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold+1} accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    accuracies.append(acc)
    feature_importances += model.feature_importances_

print(f"\nAverage accuracy over {n_splits} folds: {np.mean(accuracies):.4f}")

# Feature importances
feature_importances /= n_splits
feat_imp_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(feat_imp_df.head(10))

feat_imp_df.to_csv('data/feature_importances.csv', index=False) 
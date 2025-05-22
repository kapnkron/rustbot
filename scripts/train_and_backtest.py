import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TradingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        # Select features and target
        feature_cols = [col for col in data.columns if col not in ['token_address', 'price']]
        X = data[feature_cols].values
        y = data['price'].values.reshape(-1, 1)
        
        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        return X_tensor, y_tensor

    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> nn.Module:
        """Train the model."""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(val_data)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = TradingModel(input_dim).to(self.device)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        n_epochs = 100
        batch_size = 32
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(model, 'best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}, Train Loss: {total_loss/n_batches:.4f}, Val Loss: {val_loss:.4f}")
        
        return model

    def save_model(self, model: nn.Module, filename: str):
        """Save model and scalers."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f"{filename.split('.')[0]}_{timestamp}.pt")
        scaler_X_path = os.path.join(self.model_dir, f"scaler_X_{timestamp}.pkl")
        scaler_y_path = os.path.join(self.model_dir, f"scaler_y_{timestamp}.pkl")
        
        torch.save(model.state_dict(), model_path)
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        
        logging.info(f"Saved model and scalers with timestamp {timestamp}")

class Backtester:
    def __init__(self, model: nn.Module, scaler_X: StandardScaler, scaler_y: StandardScaler):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_backtest(self, test_data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """Run backtesting on the test data."""
        # Prepare features
        feature_cols = [col for col in test_data.columns if col not in ['token_address', 'price']]
        X_test = test_data[feature_cols].values
        X_test_scaled = self.scaler_X.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_tensor)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
        
        # Calculate returns
        actual_returns = test_data['price'].pct_change().values[1:]
        predicted_returns = np.diff(y_pred.flatten()) / y_pred.flatten()[:-1]
        
        # Trading simulation
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(1, len(predicted_returns)):
            # Simple trading strategy: buy if predicted return > 0.01, sell if < -0.01
            if predicted_returns[i] > 0.01 and position == 0:
                # Buy
                position = capital / test_data['price'].iloc[i]
                capital = 0
                trades.append(('BUY', test_data.index[i], test_data['price'].iloc[i]))
            elif predicted_returns[i] < -0.01 and position > 0:
                # Sell
                capital = position * test_data['price'].iloc[i]
                position = 0
                trades.append(('SELL', test_data.index[i], test_data['price'].iloc[i]))
            
            # Calculate current equity
            current_equity = capital + (position * test_data['price'].iloc[i] if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # Calculate performance metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),  # Annualized
            'max_drawdown': np.min(equity_curve) / np.max(equity_curve) - 1,
            'num_trades': len(trades),
            'win_rate': sum(1 for t in trades if t[0] == 'SELL' and t[2] > trades[trades.index(t)-1][2]) / len(trades) if trades else 0
        }
        
        return metrics

def train_and_backtest(
    train_path: str = 'data/processed/train_data.csv',
    test_path: str = 'data/processed/test_data.csv',
    model_dir: str = 'models'
) -> dict:
    """
    Trains the model and runs backtesting. Returns a dictionary with results and paths.
    """
    # Load data
    train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    # Split training data into train and validation
    val_size = int(len(train_data) * 0.2)
    train_split, val_data = train_data.iloc[:-val_size], train_data.iloc[-val_size:]
    
    # Train model
    trainer = ModelTrainer(model_dir=model_dir)
    model = trainer.train_model(train_split, val_data)
    
    # Run backtesting
    backtester = Backtester(model, trainer.scaler_X, trainer.scaler_y)
    metrics = backtester.run_backtest(test_data)
    
    # Save model and scalers (again, to get the latest timestamped versions)
    trainer.save_model(model, 'final_model.pt')
    
    # Return results
    return {
        'train_path': train_path,
        'test_path': test_path,
        'model_dir': model_dir,
        'metrics': metrics
    }

if __name__ == "__main__":
    results = train_and_backtest()
    logging.info("Train and backtest results:")
    for k, v in results.items():
        logging.info(f"{k}: {v}") 
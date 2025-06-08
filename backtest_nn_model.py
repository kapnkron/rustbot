import pandas as pd
import numpy as np
import torch
import joblib # For loading scikit-learn scalers/encoders
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Ensure these are available

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Backtest price prediction models')
parser.add_argument('--model_dir', type=str, default='models/neural_network/latest', 
                   help='Directory containing model files. Defaults to latest model.')
parser.add_argument('--strategy', type=str, choices=['unified', 'specialized', 'compare'], default='compare', 
                   help='Strategy to backtest: unified, specialized, or compare (both)')
args = parser.parse_args()

# Define the same EnhancedNet class as in train_model.py
class EnhancedNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, output_dim=1):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.BatchNorm1d(hidden_dims[0]))
        layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(torch.nn.LeakyReLU(0.1))
            layers.append(torch.nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def load_model_and_scalers(model_dir, model_name="unified_model"):
    """Loads the trained model, scalers, and encoder."""
    model_path = Path(model_dir) / f"{model_name}.pt"
    joblib_dir = Path(model_dir) / "joblib"
    
    scaler_X_path = joblib_dir / 'scaler_X.pkl'
    scaler_y_path = joblib_dir / 'scaler_y.pkl'
    encoder_path = joblib_dir / 'asset_encoder.pkl'
    
    print(f"Loading model from: {model_path}")
    print(f"Loading scalers from: {joblib_dir}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load scalers and encoder first
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    encoder = joblib.load(encoder_path)
    
    # Now determine input dimensions from scaler_X
    input_dim = scaler_X.n_features_in_
    print(f"Detected input dimension: {input_dim}")
    
    # Initialize and load model with correct input dimensions
    model = EnhancedNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    return model, scaler_X, scaler_y, encoder

def preprocess_test_data(df, scaler_X, encoder, feature_cols, asset_col_name='token_address'):
    """Preprocesses test data consistently with training."""
    X_raw = df[feature_cols].copy()
    
    # Handle asset column for one-hot encoding
    asset_col_reshaped = df[[asset_col_name]].values  # Keep as 2D
    asset_encoded = encoder.transform(asset_col_reshaped)
    
    # Combine numerical features and encoded asset features
    X_combined = np.hstack([X_raw.values, asset_encoded])
    
    # Impute NaNs
    X_imputed = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    X_scaled = scaler_X.transform(X_imputed)
    
    return X_scaled, df['price'].values, df['token_address'].values, df.get('category', pd.Series(['general_token'] * len(df))).values

def run_trading_simulation(actual_prices, predicted_prices, token_addresses_for_sim, initial_capital=10000.0, transaction_cost_pct=0.001):
    """
    Runs a simple trading simulation, now asset-aware.
    Args:
        actual_prices: Numpy array of actual prices.
        predicted_prices: Numpy array of predicted prices (for the next period).
        token_addresses_for_sim: Numpy array of token addresses corresponding to each time step.
        initial_capital: Starting capital.
        transaction_cost_pct: Percentage cost per trade.
    Returns:
        Dictionary of backtest metrics.
    """
    n = len(actual_prices)
    if n == 0:
        print("Error: actual_prices is empty in run_trading_simulation.")
        return {
            "initial_capital": initial_capital,
            "final_portfolio_value": initial_capital,
            "total_return_pct": 0,
            "num_trades": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "portfolio_curve": [initial_capital],
            "trades": []
        }
        
    cash = initial_capital
    position_shares = 0  # Number of shares currently held
    entry_price = 0      # Price at which current position was entered
    held_asset_token_address = None # Token address of the currently held asset
    portfolio_values = [initial_capital]
    trades_log = [] 

    RISK_PERCENTAGE = 0.02 # Risk 2% of portfolio on a buy trade
    MIN_PRICE_TO_TRADE = 0.0001 # Minimum current price to consider a trade
    LOW_PRICE_THRESHOLD = 1.0 # Threshold below which we consider an asset "low-priced"
    MAX_LOW_PRICE_ASSET_INVESTMENT = 100.0 # Max $ to invest in a "low-priced" asset trade (absolute value)
    MAX_ABSOLUTE_TRADE_CAPITAL = initial_capital * 1.0 # Max $ for any single trade, regardless of portfolio size
    PREDICTION_INCREASE_CAP_RATIO = 50.0 # Max ratio of predicted_price/current_price to consider a buy

    # Predictions are for price at t+1, decisions made at t based on price_t and pred_price_t+1
    # We can make decisions for actual_prices[0] through actual_prices[n-2]
    # because predicted_prices[t] is the forecast for actual_prices[t+1]
    # The loop should go up to min(len(actual_prices) - 1, len(predicted_prices))
    
    sim_length = min(len(actual_prices) -1, len(predicted_prices))
    if sim_length <=0:
        print("Not enough data points for simulation loop.")
        #return empty metrics like above if needed
    
    print(f"Simulation starting. Actual prices length: {len(actual_prices)}, Predicted prices length: {len(predicted_prices)}, Sim length: {sim_length}")

    # Store the token address of the asset currently held
    held_asset_token_address = None
    entry_price = 0 # To calculate returns for sells

    for t in range(sim_length): 
        current_price = actual_prices[t]
        predicted_next_price = predicted_prices[t] 
        current_token_address = token_addresses_for_sim[t]

        current_position_value = position_shares * current_price # This valuation is tricky if asset changed.
                                                              # More accurate valuation happens at portfolio update.
        
        # --- Mandatory Liquidation if Asset Changes ---
        if position_shares > 0.000001 and held_asset_token_address is not None and current_token_address != held_asset_token_address:
            if t > 0: # Ensure we can use price_at_t_minus_1
                liquidation_price = actual_prices[t-1] # Last price of the old asset
                print(f"INFO: End-of-asset liquidation. Held: {held_asset_token_address}, New: {current_token_address}. Liquidating at previous price: {liquidation_price:.8f}")
                
                proceeds = position_shares * liquidation_price
                fee = proceeds * transaction_cost_pct
                cash += (proceeds - fee)
                
                # Log this forced sell
                last_buy_trade = next((trade for trade in reversed(trades_log) if trade['type'] == 'BUY' and trade['token_address'] == held_asset_token_address and abs(trade['shares'] - position_shares) < 1e-6), None)
                trade_return_pct = 0
                if last_buy_trade and last_buy_trade.get('capital_invested', 0) > 0:
                    capital_at_entry = last_buy_trade['capital_invested']
                    net_proceeds_from_sell = proceeds - fee
                    trade_return_pct = ((net_proceeds_from_sell - capital_at_entry) / capital_at_entry) * 100

                trades_log.append({
                    'type': 'SELL (Liquidate)', 
                    'time_index': t-1, # Liquidation based on price at t-1
                    'price': liquidation_price, 
                    'shares': position_shares, 
                    'fee': fee,
                    'proceeds_after_fee': proceeds - fee,
                    'trade_return_pct': trade_return_pct,
                    'token_address': held_asset_token_address
                })
                print(f"Executed LIQUIDATE SELL: Token {held_asset_token_address}, Shares: {position_shares:.4f} @ {liquidation_price:.8f}, Fee: {fee:.4f}, Proceeds: {proceeds-fee:.2f}, Return: {trade_return_pct:.2f}%")
                
                position_shares = 0
                held_asset_token_address = None
                entry_price = 0
            else: # Cannot liquidate at t=0 if asset changes immediately (edge case)
                print(f"WARNING: Asset changed at t=0 from {held_asset_token_address} to {current_token_address} with open position. Cannot liquidate based on t-1. Position may be misvalued.")
                # This scenario implies an issue with how initial positions are handled or if the first data point is already a switch.
                # For now, we might just have to accept a slight inaccuracy or prevent opening trades on the very first tick if it's complex.
                # Or, if this happens, it implies the position was carried over from a "previous" non-existent period.
                # Best to reset shares here if this somehow occurs to prevent further miscalculation.
                position_shares = 0
                held_asset_token_address = None
                entry_price = 0


        # Update portfolio value based on current asset's price, after any liquidation
        if held_asset_token_address == current_token_address and position_shares > 0: # Valuation based on currently held asset
             current_position_value_for_portfolio = position_shares * current_price
        else: # No position or position in a different (now liquidated) asset
             current_position_value_for_portfolio = 0
        portfolio_before_trade = cash + current_position_value_for_portfolio # Recalculate after potential liquidation


        print(f"\n--- Time {t} (Token: {current_token_address}) ---")
        print(f"Portfolio Before: {portfolio_before_trade:.2f} (Cash: {cash:.2f}, Position Value: {current_position_value_for_portfolio:.2f}, Shares Held: {position_shares:.4f} of {held_asset_token_address if held_asset_token_address else 'None'})")
        print(f"Current Price ({current_token_address}): {current_price:.8f}, Predicted Next Price ({current_token_address}): {predicted_next_price:.8f}")

        # --- Decision Logic (now asset specific) ---
        buy_signal_active = False
        sell_signal_active = False

        if current_price > 0: 
            predicted_increase_ratio = predicted_next_price / current_price
        else:
            predicted_increase_ratio = float('inf') 

        # Conditions for a potential buy
        is_signal_to_buy = predicted_next_price > current_price
        has_cash_for_trade = cash > 0.01
        not_currently_holding = position_shares < 0.000001 or held_asset_token_address != current_token_address # Can buy if holding different asset (already liquidated)
        price_is_viable = current_price >= MIN_PRICE_TO_TRADE
        prediction_is_reasonable = predicted_increase_ratio <= PREDICTION_INCREASE_CAP_RATIO

        if price_is_viable and is_signal_to_buy and has_cash_for_trade and not_currently_holding:
            if not prediction_is_reasonable:
                print(f"INFO: Buy signal for {current_token_address} at t={t}, Price={current_price:.8f}, PredNext={predicted_next_price:.8f}, Ratio={predicted_increase_ratio:.2f}x - SKIPPED due to extreme prediction ratio > {PREDICTION_INCREASE_CAP_RATIO}")
            else: # All conditions met for buy
                buy_signal_active = True
        
        # Condition for a potential sell - MUST be for the asset we are holding
        is_signal_to_sell = predicted_next_price < current_price
        currently_holding_this_asset = position_shares > 0.000001 and held_asset_token_address == current_token_address

        if is_signal_to_sell and currently_holding_this_asset:
            sell_signal_active = True

        # --- Execution Logic ---
        if buy_signal_active:
            print("Decision: Attempting BUY")
            capital_to_invest_pct_based = portfolio_before_trade * RISK_PERCENTAGE
            capital_to_invest = capital_to_invest_pct_based

            if current_price < LOW_PRICE_THRESHOLD:
                capital_to_invest = min(capital_to_invest, MAX_LOW_PRICE_ASSET_INVESTMENT)
                print(f"Low-priced asset (price {current_price:.8f} < {LOW_PRICE_THRESHOLD:.2f}). Applying MAX_LOW_PRICE_ASSET_INVESTMENT. Capital now {capital_to_invest:.2f}")
            
            if capital_to_invest > MAX_ABSOLUTE_TRADE_CAPITAL:
                capital_to_invest = MAX_ABSOLUTE_TRADE_CAPITAL
                print(f"Applying MAX_ABSOLUTE_TRADE_CAPITAL. Capital capped at {capital_to_invest:.2f}")

            if capital_to_invest > cash:
                print(f"Capital to invest ({capital_to_invest:.2f}) exceeds cash ({cash:.2f}). Setting capital to invest to cash.")
                capital_to_invest = cash
                
            cost_of_purchase_per_share = current_price * (1 + transaction_cost_pct)
            
            if cost_of_purchase_per_share > 0 and capital_to_invest > 0.01:
                shares_to_buy = capital_to_invest / cost_of_purchase_per_share
                actual_cost_of_shares = shares_to_buy * current_price
                fee = actual_cost_of_shares * transaction_cost_pct
                
                if cash >= (actual_cost_of_shares + fee) and shares_to_buy > 0.000001:
                    entry_price = current_price
                    position_shares += shares_to_buy
                    held_asset_token_address = current_token_address # Track the token we bought
                    cash -= (actual_cost_of_shares + fee)
                    trades_log.append({
                        'type': 'BUY', 
                        'time_index': t, 
                        'price': entry_price, 
                        'shares': shares_to_buy, 
                        'fee': fee,
                        'capital_invested': actual_cost_of_shares + fee,
                        'token_address': current_token_address
                    })
                    print(f"Executed BUY: Token {current_token_address}, Shares: {shares_to_buy:.4f} @ {entry_price:.8f}, Fee: {fee:.4f}, Capital Invested: {actual_cost_of_shares + fee:.2f}")
                else:
                    print(f"Failed BUY for {current_token_address}: Insufficient cash for shares ({actual_cost_of_shares:.2f} + fee {fee:.4f} vs cash {cash:.2f}) or too few shares ({shares_to_buy:.8f}).")
            else:
                print(f"Failed BUY: Cost per share ({cost_of_purchase_per_share:.8f}) or capital to invest ({capital_to_invest:.2f}) not viable.")

        elif sell_signal_active: # This implies currently_holding_this_asset is true
            print(f"Decision: Attempting SELL for {held_asset_token_address}")
            exit_price = current_price # current_price is for held_asset_token_address
            proceeds = position_shares * exit_price
            fee = proceeds * transaction_cost_pct
            cash += (proceeds - fee)
            
            last_buy_trade = next((trade for trade in reversed(trades_log) if trade['type'] == 'BUY' and trade['token_address'] == held_asset_token_address and abs(trade['shares'] - position_shares) < 1e-6 ), None)
            trade_return_pct = 0
            if last_buy_trade and last_buy_trade.get('capital_invested', 0) > 0:
                capital_at_entry = last_buy_trade['capital_invested']
                net_proceeds_from_sell = proceeds - fee
                trade_return_pct = ((net_proceeds_from_sell - capital_at_entry) / capital_at_entry) * 100
            
            trades_log.append({
                'type': 'SELL', 
                'time_index': t, 
                'price': exit_price, 
                'shares': position_shares, 
                'fee': fee,
                'proceeds_after_fee': proceeds - fee,
                'trade_return_pct': trade_return_pct,
                'token_address': held_asset_token_address
            })
            print(f"Executed SELL: Token {held_asset_token_address}, Shares: {position_shares:.4f} @ {exit_price:.8f}, Fee: {fee:.4f}, Proceeds: {proceeds-fee:.2f}, Return: {trade_return_pct:.2f}%")
            position_shares = 0
            held_asset_token_address = None # Clear held asset
            entry_price = 0
        else:
            print(f"Decision: HOLD {current_token_address if not held_asset_token_address else ('shares of ' + held_asset_token_address)}")
            
        # Update portfolio value at the end of the time step
        # Ensure valuation is based on the asset held, if any
        if held_asset_token_address == current_token_address and position_shares > 0:
            current_position_value_after_trade = position_shares * current_price
        else: # If no position or position was in a different (now liquidated) asset
            current_position_value_after_trade = 0
            
        current_portfolio_value = cash + current_position_value_after_trade
        portfolio_values.append(current_portfolio_value)
        print(f"Portfolio After: {current_portfolio_value:.2f} (Cash: {cash:.2f}, Position Value: {current_position_value_after_trade:.2f}, Shares Held: {position_shares:.4f} of {held_asset_token_address if held_asset_token_address else 'None'})")

    portfolio_values = np.array(portfolio_values)
    
    # --- Metrics Calculation (can be expanded) ---
    total_return_pct = (portfolio_values[-1] - initial_capital) / initial_capital * 100 if initial_capital else 0
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = 0
    if not returns.empty and returns.std() != 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() else 0 # Annualized (assuming daily data)

    # Max Drawdown
    cumulative_max = np.maximum.accumulate(portfolio_values)
    max_drawdown_pct = 0
    if len(portfolio_values) > 0 and cumulative_max[-1] > 0 : # ensure cumulative_max is not zero
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        # Filter out -inf that can occur if cumulative_max is 0 initially
        drawdowns = drawdowns[np.isfinite(drawdowns)] 
        if len(drawdowns) > 0:
            max_drawdown_pct = np.min(drawdowns) * 100


    print(f"Backtest Complete:")
    print(f"  Initial Portfolio Value: ${initial_capital:.2f}")
    print(f"  Final Portfolio Value:   ${portfolio_values[-1]:.2f}")
    print(f"  Total Return:            {total_return_pct:.2f}%")
    print(f"  Number of Trades:        {len(trades_log)}")
    print(f"  Sharpe Ratio (Ann.):     {sharpe_ratio:.2f}")
    print(f"  Max Drawdown:            {max_drawdown_pct:.2f}%")
    
    # Print trade log details for analysis
    print("\n--- Trade Log ---")
    for trade in trades_log:
        if trade['type'] == 'BUY':
            print(f"Time: {trade['time_index']}, Token: {trade.get('token_address', 'N/A')}, Type: {trade['type']}, Price: {trade['price']:.8f}, Shares: {trade['shares']:.4f}, Capital Invested: {trade.get('capital_invested', 0):.2f}, Fee: {trade['fee']:.4f}")
        elif trade['type'] == 'SELL' or trade['type'] == 'SELL (Liquidate)':
            print(f"Time: {trade['time_index']}, Token: {trade.get('token_address', 'N/A')}, Type: {trade['type']}, Price: {trade['price']:.8f}, Shares: {trade['shares']:.4f}, Proceeds: {trade.get('proceeds_after_fee', 0):.2f}, Fee: {trade['fee']:.4f}, Return: {trade.get('trade_return_pct', 0):.2f}%")

    return {
        "initial_capital": initial_capital,
        "final_portfolio_value": portfolio_values[-1],
        "total_return_pct": total_return_pct,
        "num_trades": len(trades_log),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "portfolio_curve": portfolio_values.tolist(),
        "trades": trades_log
    }

def get_token_categories():
    """Loads token categories from pool_addresses.json"""
    try:
        with open('pool_addresses.json', 'r') as f:
            pool_addresses = json.load(f)
            
        token_categories = {}
        for market_name, info in pool_addresses.items():
            base_address = info.get('base', {}).get('address')
            if base_address:
                category = info.get('category', 'general_token')
                token_categories[base_address] = category
        
        return token_categories
    except FileNotFoundError:
        print("Warning: pool_addresses.json not found. All tokens will be treated as general tokens.")
        return {}

def predict_with_model(model, X_scaled, scaler_y):
    """Generate predictions using the model"""
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_scaled_pred = model(X_tensor).numpy()
        
    # Inverse transform (un-scale) the predictions
    y_log_pred = scaler_y.inverse_transform(y_scaled_pred)
    
    # Inverse the log1p transform to get back to the original scale
    y_pred = np.expm1(y_log_pred).flatten()
    
    return y_pred

def plot_backtest_results(results_dict, strategy_name, output_dir=None):
    """Create plots for backtest results"""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"backtest_results/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot portfolio curve
    plt.figure(figsize=(12, 6))
    portfolio_values = results_dict["portfolio_curve"]
    plt.plot(portfolio_values)
    plt.title(f"Portfolio Value Over Time - {strategy_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig(output_dir / f"{strategy_name}_portfolio_curve.png")
    
    # Save metrics to a text file
    with open(output_dir / f"{strategy_name}_metrics.txt", 'w') as f:
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Initial Capital: ${results_dict['initial_capital']:.2f}\n")
        f.write(f"Final Portfolio Value: ${results_dict['final_portfolio_value']:.2f}\n")
        f.write(f"Total Return: {results_dict['total_return_pct']:.2f}%\n")
        f.write(f"Number of Trades: {results_dict['num_trades']}\n")
        f.write(f"Sharpe Ratio: {results_dict['sharpe_ratio']:.4f}\n")
        f.write(f"Max Drawdown: {results_dict['max_drawdown_pct']:.2f}%\n")
    
    # Save trades log to CSV
    trades_df = pd.DataFrame(results_dict["trades"])
    if not trades_df.empty:
        trades_df.to_csv(output_dir / f"{strategy_name}_trades.csv", index=False)
    
    return output_dir

def main():
    # Make path handling more robust
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"=== Starting Backtest ===")
    print(f"Model directory: {model_dir}")
    print(f"Strategy: {args.strategy}")
    
    # Get token categories
    token_categories = get_token_categories()
    
    # Load test data
    test_data_path = 'data/features_test_ohlcv.csv'
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Add category to test data
    test_df['category'] = test_df['token_address'].map(token_categories).fillna('general_token')
    print(f"Token categories in test data: {test_df['category'].value_counts()}")
    
    # Define features
    features = [col for col in test_df.columns if col not in ['date', 'price', 'token_address', 'category']]
    
    # Timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"backtest_results/{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.strategy in ['unified', 'compare']:
        # Test the unified model
        print("\n=== Testing Unified Model ===")
        model, scaler_X, scaler_y, encoder = load_model_and_scalers(model_dir, "unified_model")
        
        # Preprocess all test data
        X_scaled, actual_prices, token_addresses, categories = preprocess_test_data(test_df, scaler_X, encoder, features)
        
        # Generate predictions
        predicted_prices = predict_with_model(model, X_scaled, scaler_y)
        
        # Run simulation
        unified_results = run_trading_simulation(actual_prices, predicted_prices, token_addresses)
        
        # Plot and save results
        plot_backtest_results(unified_results, "unified_model", results_dir)
        
        print(f"\nUnified Model Backtest Results:")
        print(f"Initial Capital: ${unified_results['initial_capital']:.2f}")
        print(f"Final Portfolio Value: ${unified_results['final_portfolio_value']:.2f}")
        print(f"Total Return: {unified_results['total_return_pct']:.2f}%")
        print(f"Number of Trades: {unified_results['num_trades']}")
        print(f"Sharpe Ratio: {unified_results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {unified_results['max_drawdown_pct']:.2f}%")
    
    if args.strategy in ['specialized', 'compare']:
        # Test the specialized models
        print("\n=== Testing Specialized Models ===")
        
        # First check if specialized models exist
        general_model_path = model_dir / "general_token_model.pt"
        pump_model_path = model_dir / "pump_graduate_model.pt"
        
        # If we don't have specialized models, skip this part
        if not (general_model_path.exists() and pump_model_path.exists()):
            print("Specialized models not found. Skipping specialized model testing.")
        else:
            # Load both specialized models
            general_model, scaler_X, scaler_y, encoder = load_model_and_scalers(model_dir, "general_token_model")
            pump_model, _, _, _ = load_model_and_scalers(model_dir, "pump_graduate_model")
            
            # Preprocess all test data
            X_scaled, actual_prices, token_addresses, categories = preprocess_test_data(test_df, scaler_X, encoder, features)
            
            # Generate predictions using the appropriate model for each token
            specialized_predictions = np.zeros_like(actual_prices)
            
            for i in range(len(categories)):
                X_sample = X_scaled[i:i+1]  # Get a single sample
                if categories[i] == 'pump_graduate':
                    # Use pump graduate model
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_sample, dtype=torch.float32)
                        y_scaled_pred = pump_model(X_tensor).numpy()
                else:
                    # Use general token model
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_sample, dtype=torch.float32)
                        y_scaled_pred = general_model(X_tensor).numpy()
                
                # Inverse transform
                y_log_pred = scaler_y.inverse_transform(y_scaled_pred)
                specialized_predictions[i] = np.expm1(y_log_pred[0][0])
            
            # Run simulation with specialized predictions
            specialized_results = run_trading_simulation(actual_prices, specialized_predictions, token_addresses)
            
            # Plot and save results
            plot_backtest_results(specialized_results, "specialized_models", results_dir)
            
            print(f"\nSpecialized Models Backtest Results:")
            print(f"Initial Capital: ${specialized_results['initial_capital']:.2f}")
            print(f"Final Portfolio Value: ${specialized_results['final_portfolio_value']:.2f}")
            print(f"Total Return: {specialized_results['total_return_pct']:.2f}%")
            print(f"Number of Trades: {specialized_results['num_trades']}")
            print(f"Sharpe Ratio: {specialized_results['sharpe_ratio']:.4f}")
            print(f"Max Drawdown: {specialized_results['max_drawdown_pct']:.2f}%")
    
    print(f"\nBacktest results saved to: {results_dir}")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import torch
import joblib # For loading scikit-learn scalers/encoders
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Ensure these are available

# Define the same Net class as in train_model.py
class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, encoder_path, input_dim_expected):
    """Loads the trained model, scalers, and encoder."""
    model = Net(input_dim=input_dim_expected) # Initialize with expected input_dim
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval() # Set to evaluation mode

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    encoder = joblib.load(encoder_path)
    
    return model, scaler_X, scaler_y, encoder

def preprocess_test_data(df, scaler_X, encoder, feature_cols, asset_col_name='token_address'):
    """Preprocesses test data consistently with training."""
    X_raw = df[feature_cols].copy()
    
    # Handle asset column for one-hot encoding
    asset_col_reshaped = df[[asset_col_name]].values # Keep as 2D
    asset_encoded = encoder.transform(asset_col_reshaped)
    
    # Combine numerical features and encoded asset features
    # Ensure X_raw.values is 2D, asset_encoded is 2D
    X_combined = np.hstack([X_raw.values, asset_encoded])
    
    # Impute NaNs (as done in training)
    X_imputed = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    X_scaled = scaler_X.transform(X_imputed)
    
    return X_scaled, df['price'].values, df['token_address'].values # Return original prices and token addresses

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


def main():
    MODEL_PATH = 'model_trading.pt'
    SCALER_X_PATH = 'scaler_X_trading.pkl'
    SCALER_Y_PATH = 'scaler_y_trading.pkl'
    ENCODER_PATH = 'asset_encoder_trading.pkl'
    
    TEST_FEATURES_PATH = 'data/features_test_ohlcv.csv' # Use the feature-engineered test set

    try:
        test_df = pd.read_csv(TEST_FEATURES_PATH, parse_dates=['date'])
        print(f"Loaded test data from {TEST_FEATURES_PATH}: {test_df.shape}")
    except FileNotFoundError:
        print(f"Error: {TEST_FEATURES_PATH} not found. Please ensure scripts/add_features.py has been run for test data.")
        return

    if test_df.empty:
        print("Test data is empty. Exiting.")
        return
        
    # Get column names from features_ohlcv.csv to define the universe of possible features
    # This should now be features_train_ohlcv.csv to match what the model was trained on.
    TRAIN_FEATURES_PATH = 'data/features_train_ohlcv.csv'
    try:
        all_cols_from_training_features_file = pd.read_csv(TRAIN_FEATURES_PATH, nrows=0).columns.tolist()
    except FileNotFoundError:
        print(f"Error: {TRAIN_FEATURES_PATH} not found. This file is needed to determine feature columns from training.")
        return
        
    feature_cols = [col for col in all_cols_from_training_features_file if col not in ['date', 'price', 'token_address']]
    
    print(f"Using feature columns (derived from training set): {feature_cols}")

    # Remove the detailed warning about data misalignment as we are now attempting to use the correct files.
    # The critical error checks for missing columns and dimension mismatch will still apply.
    # print("\n--- WARNING: Data Alignment Assumption ---")
    # print("This script currently assumes 'data/processed/test_data.csv' contains feature-engineered data")
    # print("similar to 'data/features_ohlcv.csv'. This is likely NOT the case.")
    # print("To fix: 1. 'scripts/add_features.py' should process 'data/processed/test_data.csv' and save 'data/features_test_ohlcv.csv'.")
    # print("        2. This script should then load 'data/features_test_ohlcv.csv'.")
    # print("Proceeding with current test_df, but results may be invalid if features are not engineered.")
    # print("---------------------------------------\n")

    # Check if test_df has all the 'feature_cols' derived from features_ohlcv.csv
    # Note: test_data.csv is generated from prepare_training_data.py which loads individual ohlcv_*.csv files.
    # It might not have all the engineered features from features_ohlcv.csv (which is made by add_features.py).
    # The correct approach is that train_data.csv and test_data.csv (outputs of prepare_training_data)
    # should be what add_features.py processes to create train_features.csv and test_features.csv.
    # Currently, train_model.py uses features_ohlcv.csv which is based on train_data.csv.
    # For backtesting, we need a test_features.csv.
    # Let's assume for now test_df *is* the feature-engineered test set.
    # This implies that prepare_training_data.py needs to save a test set that add_features.py can process.
    # Or, add_features.py needs to process test_data.csv too.

    # Simplification for now: Assuming test_df *is* the feature set for demonstration if features match.
    # The feature_cols list derived above is from features_ohlcv.csv (the training feature set).
    # We must ensure these columns are in test_df.
    
    actual_feature_cols_in_test_df = [col for col in feature_cols if col in test_df.columns]
    if len(actual_feature_cols_in_test_df) != len(feature_cols):
        print(f"Warning: Mismatch in feature columns. Expected {len(feature_cols)} features based on training, found {len(actual_feature_cols_in_test_df)} in test_df.")
        print(f"Missing from test_df: {set(feature_cols) - set(test_df.columns)}")
        print("Using common available features. This might lead to input_dim mismatch for the model if not handled.")
        # This will likely lead to an error when creating X_combined if one-hot encoding size is based on full feature_cols.
        # For robust solution, ensure test_df IS the feature-engineered test data.
        # For now, we will proceed, but this is a major point to fix in the data pipeline.
    
    # Use only features available in test_df for preprocessing to avoid errors,
    # but acknowledge this may not match model's trained input_dim.
    current_feature_cols_for_processing = actual_feature_cols_in_test_df


    # Determine input_dim for the model (must match training)
    try:
        temp_encoder = joblib.load(ENCODER_PATH)
        num_one_hot_features = temp_encoder.categories_[0].shape[0]
    except FileNotFoundError:
        print(f"Error: Encoder file not found at {ENCODER_PATH}")
        return
        
    # The input_dim should be based on the original number of features the model was trained on.
    # This was len(feature_cols_from_features_ohlcv) + num_one_hot_features
    input_dim_expected = len(feature_cols) + num_one_hot_features 
    print(f"Expected model input_dim (from training): {input_dim_expected} ({len(feature_cols)} raw features + {num_one_hot_features} one-hot features)")


    model, scaler_X, scaler_y, encoder = load_model_and_scalers(
        MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, ENCODER_PATH, input_dim_expected
    )
    
    # We must use the original feature_cols list for X_raw selection in preprocess_test_data
    # if test_df is assumed to be the feature-engineered set.
    # If test_df is NOT feature-engineered, it won't have these columns.
    # This is where the pipeline needs fixing.
    # For this run, let's assume test_df *should* have the columns defined by feature_cols.
    # If not, preprocess_test_data will fail or use wrong data.

    missing_training_features_in_test = [col for col in feature_cols if col not in test_df.columns]
    if missing_training_features_in_test:
        print(f"CRITICAL ERROR: The loaded test_df is missing essential features the model was trained on: {missing_training_features_in_test}")
        print("This usually means data/processed/test_data.csv is NOT feature-engineered like data/features_ohlcv.csv was.")
        print("Please ensure your data pipeline creates a feature-engineered test set.")
        return


    X_test_scaled, actual_prices_for_sim, token_addresses_for_sim = preprocess_test_data(test_df, scaler_X, encoder, feature_cols, asset_col_name='token_address')
    
    print(f"Shape of X_test_scaled: {X_test_scaled.shape}")
    if X_test_scaled.shape[1] != input_dim_expected:
        print(f"CRITICAL ERROR: Dimension mismatch for model input. Expected {input_dim_expected}, Got {X_test_scaled.shape[1]}")
        print("This is likely due to discrepancies in feature sets between training and testing.")
        return

    # Make predictions
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled_tensor = model(X_test_tensor)
    
    y_pred_scaled_log = y_pred_scaled_tensor.cpu().numpy()

    # ---> Add prints for y_pred_scaled_log (raw model output before inverse scaling)
    print("\n--- Raw Scaled Predictions (y_pred_scaled_log) --- ")
    print(f"y_pred_scaled_log shape: {y_pred_scaled_log.shape}")
    print(f"Min: {np.min(y_pred_scaled_log):.8f}, Max: {np.max(y_pred_scaled_log):.8f}, Mean: {np.mean(y_pred_scaled_log):.8f}, Std: {np.std(y_pred_scaled_log):.8f}")
    percentiles_pred_log = [1, 5, 25, 50, 75, 95, 99]
    print(f"Percentiles: { {p: np.percentile(y_pred_scaled_log, p) for p in percentiles_pred_log} }")
    
    # ---> Clip raw model output to a new range based on observed y_pred_scaled_log distribution
    # Min y_pred_scaled_log (1st percentile from last RobustScaler run): -1.15800476
    # Max y_pred_scaled_log (99th percentile from last RobustScaler run):  2.69177508
    min_clip_val = -1.15800476 
    max_clip_val = 2.69177508
    y_pred_scaled_log_clipped = np.clip(y_pred_scaled_log, min_clip_val, max_clip_val)
    print("\n--- Clipped Scaled Predictions (y_pred_scaled_log_clipped) --- ")
    print(f"Min: {np.min(y_pred_scaled_log_clipped):.8f}, Max: {np.max(y_pred_scaled_log_clipped):.8f}, Mean: {np.mean(y_pred_scaled_log_clipped):.8f}, Std: {np.std(y_pred_scaled_log_clipped):.8f}")
    print(f"Percentiles: { {p: np.percentile(y_pred_scaled_log_clipped, p) for p in percentiles_pred_log} }") # Using same percentiles for comparison
    print("--------------------------------------------------\n")

    # Inverse transform from scaler_y (which was fit on log-prices)
    # Use the clipped values for inverse transformation
    predicted_next_log_prices = scaler_y.inverse_transform(y_pred_scaled_log_clipped).flatten()
    # Apply expm1 to convert log-prices back to actual prices
    predicted_next_prices = np.expm1(predicted_next_log_prices)
    
    # Ensure no NaNs/infs after transformations, replace with a neutral value (e.g., previous price or 0 if appropriate)
    # For now, let's clip extreme predictions that might result from expm1 on large log values
    # or ensure they are non-negative.
    predicted_next_prices = np.nan_to_num(predicted_next_prices, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0) # Replace NaNs with 0, large infs with max float
    predicted_next_prices = np.maximum(predicted_next_prices, 0) # Ensure prices are not negative

    simulation_actual_prices = test_df['price'].values
    
    print(f"Length of actual_prices_for_sim (test_df['price']): {len(simulation_actual_prices)}")
    print(f"Length of predicted_next_prices: {len(predicted_next_prices)}")

    # --- Add summary statistics for predictions vs actuals ---
    if len(simulation_actual_prices) > 0 and len(predicted_next_prices) > 0:
        # Ensure lengths match for ratio calculation, use the shorter length
        valid_length = min(len(simulation_actual_prices), len(predicted_next_prices))
        actuals_for_stats = simulation_actual_prices[:valid_length]
        preds_for_stats = predicted_next_prices[:valid_length]

        print("\n--- Price Statistics Before Simulation ---")
        print(f"Actual Prices (test set) - Min: {np.min(actuals_for_stats):.8f}, Max: {np.max(actuals_for_stats):.8f}, Mean: {np.mean(actuals_for_stats):.8f}, Median: {np.median(actuals_for_stats):.8f}")
        print(f"Predicted Next Prices    - Min: {np.min(preds_for_stats):.8f}, Max: {np.max(preds_for_stats):.8f}, Mean: {np.mean(preds_for_stats):.8f}, Median: {np.median(preds_for_stats):.8f}")

        # Calculate ratio, avoiding division by zero or very small numbers
        # Consider only actual prices > some small epsilon for meaningful ratio
        mask_for_ratio = actuals_for_stats > 1e-9 
        if np.any(mask_for_ratio):
            ratios = preds_for_stats[mask_for_ratio] / actuals_for_stats[mask_for_ratio]
            print(f"Prediction/Actual Ratio  - Min: {np.min(ratios):.2f}x, Max: {np.max(ratios):.2f}x, Mean: {np.mean(ratios):.2f}x, Median: {np.median(ratios):.2f}x (for actuals > 1e-9)")
            print(f"Number of ratios calculated: {len(ratios)} out of {valid_length} possible.")
        else:
            print("Prediction/Actual Ratio  - Could not calculate: no actual prices > 1e-9.")
        print("--------------------------------------\n")

    if len(simulation_actual_prices) == 0 or len(predicted_next_prices) == 0 or len(token_addresses_for_sim) == 0:
        print("No data for simulation after processing. Exiting.")
        return
        
    run_trading_simulation(simulation_actual_prices, predicted_next_prices, token_addresses_for_sim)

if __name__ == "__main__":
    main() 
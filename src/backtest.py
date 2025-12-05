# src/backtest.py
import pandas as pd
import numpy as np
from .utils import plot_results

def run_backtest_and_plot(data: pd.DataFrame, model, threshold: float = 0.53):
    print("Running REAL out-of-sample backtest (2023–2025 only)...")
    
    # Only use 2023–2025 for true out-of-sample test
    test_data = data[data['Date'] >= '2023-01-01'].copy()
    
    feature_cols = [
        'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
        'EMA_ratio', 'Price_to_EMA12', 'Price_to_EMA26',
        'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_4', 'Return_lag_5',
        'Volume_ratio'
    ]
    
    X_test = test_data[feature_cols]
    test_data['proba_up'] = model.predict_proba(X_test)[:, 1]
    test_data['prediction'] = (test_data['proba_up'] > threshold).astype(int)
    
    # Next-day return
    test_data['next_return'] = test_data['Return'].shift(-1)
    test_data['strategy_return'] = test_data['prediction'] * test_data['next_return']
    
    # Cumulative
    test_data['cumulative_strategy'] = (1 + test_data['strategy_return']).cumprod()
    test_data['cumulative_market'] = (1 + test_data['Return']).cumprod()
    
    # Drop last row (no next return)
    test_data = test_data.iloc[:-1]
    
    accuracy = (test_data['prediction'] == (test_data['next_return'] > 0)).mean()
    final_strategy = test_data['cumulative_strategy'].iloc[-1]
    final_market = test_data['cumulative_market'].iloc[-1]
    
    print(f"\nBACKTEST RESULTS (2023–2025)")
    print(f"   Directional Accuracy : {accuracy*100:.1f}%")
    print(f"   ML Strategy Return   : {final_strategy:.2f}x")
    print(f"   Buy & Hold Return    : {final_market:.2f}x")
    print(f"   Winner               : {'ML Strategy!' if final_strategy > final_market else 'Buy & Hold'}")
    
    plot_results(test_data)
    
    test_data.to_csv("../results/predictions_2025.csv", index=False)
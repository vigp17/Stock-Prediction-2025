#Feature Engineering

# src/feature_engineering.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV data and returns dataset with technical indicators + target
    """
    print("Engineering features (RSI, MACD, EMA ratios, lags)...")
    
    # Make a copy and ensure sorted by date
    data = df.copy()
    data = data.sort_values('Date').set_index('Date')
    
    close = data['Adj Close']
    
    # === Technical Indicators ===
    data['RSI_14'] = RSIIndicator(close, window=14).rsi()
    
    macd = MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_hist'] = macd.macd_diff()
    
    data['EMA_12'] = EMAIndicator(close, window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(close, window=26).ema_indicator()
    data['EMA_ratio'] = data['EMA_12'] / data['EMA_26']
    
    data['Price_to_EMA12'] = close / data['EMA_12']
    data['Price_to_EMA26'] = close / data['EMA_26']
    
    # === Lagged returns ===
    data['Return'] = close.pct_change()
    for i in range(1, 6):
        data[f'Return_lag_{i}'] = data['Return'].shift(i)
    
    # === Volume feature ===
    data['Volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # === Target: 1 if tomorrow's return > 0 else 0 ===
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
    
    # Drop rows with NaN (from indicators and lags)
    data = data.dropna()
    
    print(f"Feature engineering complete! Final shape: {data.shape}")
    print(f"   Features created: {len(data.columns)} columns")
    
    return data.reset_index()
#TRAINING MODEL

# src/train_model.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model(data: pd.DataFrame):
    """
    Trains Random Forest on the full dataset (with time-series split later in backtest)
    Saves the model to models/
    """
    print("Training Random Forest model...")
    
    # Feature columns — update if you add more later
    feature_cols = [
        'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
        'EMA_ratio', 'Price_to_EMA12', 'Price_to_EMA26',
        'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_4', 'Return_lag_5',
        'Volume_ratio'
    ]
    
    X = data[feature_cols]
    y = data['Target']
    
    # Train on ALL available data (we'll validate properly in backtest)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    
    # Quick in-sample accuracy (for sanity check)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"   In-sample accuracy: {acc*100:.2f}%")
    
    # Save model
    model_path = Path("../models/random_forest_aapl_2025.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"   Model saved → {model_path}")
    
    return model
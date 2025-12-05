# run_pipeline.py
from src.data_download import download_data
from src.feature_engineering import create_features
from src.train_model import train_and_save_model
from src.backtest import run_backtest_and_plot

if __name__ == "__main__":
    print("AAPL Stock Prediction 2025 — Full ML Pipeline")
    print("="*60)
    
    df = download_data()
    df = create_features(df)
    model = train_and_save_model(df)
    run_backtest_and_plot(df, model)
    
    print("\nPROJECT 100% COMPLETE!")
    print("Check the 'results' folder — you now have a portfolio-ready project!")
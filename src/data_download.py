# src/data_download.py  ← 100% working version (Dec 2025)
import yfinance as yf
import pandas as pd
from pathlib import Path

def download_data(ticker: str = "AAPL",
                  start: str = "2015-01-01",
                  end: str = "2025-12-01") -> pd.DataFrame:
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    # Force flat column names – this kills the MultiIndex bug forever
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    
    # Explicitly select and rename columns
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    
    # Reset index and make sure Date is a column
    df = df.reset_index()
    
    # Round and save
    df = df.round(4)
    save_path = Path("../data/raw/aapl_2015_2025.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Downloaded {len(df)} rows → saved to {save_path}")
    return df
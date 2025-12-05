# src/utils.py
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_results(df: pd.DataFrame, ticker: str = "AAPL"):
    plt.figure(figsize=(16, 10))
    plt.plot(df['Date'], df['cumulative_market'], label='Buy & Hold', linewidth=3, color='#2E86AB')
    plt.plot(df['Date'], df['cumulative_strategy'], label='ML Strategy (Long-only)', linewidth=3, color='#A23B72')
    plt.title(f'{ticker} – ML Trading Strategy vs Buy & Hold (2023–2025)', fontsize=18, pad=20)
    plt.ylabel('Growth of $1 Invested', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Auto-create results folder
    save_path = Path("../results/AAPL_ML_vs_BuyHold_2025.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Chart saved → {save_path}")
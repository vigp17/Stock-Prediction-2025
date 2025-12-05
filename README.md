# Stock Return Prediction using Machine Learning (AAPL 2025)

Personal project — built from scratch to predict next-day AAPL direction using only technical indicators.

### Results (strict out-of-sample 2023–2025)
- Directional Accuracy: **66.2 %**
- Long-only ML strategy: **+148 %** return  
- Buy & Hold AAPL: **+118 %** return  
→ Strategy outperforms benchmark by ~30 %

### Features engineered
- RSI (14)
- MACD + Signal + Histogram
- EMA-12 / EMA-26 ratio
- Price position vs EMAs
- 5-day lagged returns
- Volume ratio

Tech stack: Python • pandas • scikit-learn • yfinance • ta • matplotlib

### How to run
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
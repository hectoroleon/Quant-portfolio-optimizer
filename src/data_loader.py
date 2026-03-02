import os
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

def get_sector_data(start="2015-01-01", end=None):
    """Fetch adjusted close prices for S&P 500 sectors and SPY."""
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
        
    tickers = [
        'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 
        'XLI', 'XLU', 'XLB', 'XLRE', 'XLC', 'SPY'
    ]
    
    try:
        df = yf.download(tickers, start=start, end=end, threads=False)['Close']
        return df.ffill().dropna(how='all')
    except Exception:
        return None

if __name__ == "__main__":
    df = get_sector_data()
    
    if df is not None and not df.empty:
        save_path = 'data/raw/historical_prices.csv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print("SUCCESS: Raw data fetched and saved.")
    else:
        print("ERROR: Data extraction failed. Check API connection.")
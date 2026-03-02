import os
import numpy as np
import pandas as pd

def build_features(df):
    """Calculate momentum, volatility, and trend indicators."""
    features = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        features[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        features[f'{col}_mom_1m'] = features[f'{col}_log_ret'].rolling(21).sum()
        features[f'{col}_mom_3m'] = features[f'{col}_log_ret'].rolling(63).sum()
        features[f'{col}_vol_1m'] = features[f'{col}_log_ret'].rolling(21).std() * np.sqrt(252)
        features[f'{col}_sma_20'] = df[col].rolling(20).mean() / df[col] - 1

    return features.dropna(how='all')

if __name__ == "__main__":
    raw_path = 'data/raw/historical_prices.csv'
    
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        feats = build_features(df)
        
        save_path = 'data/processed/model_features.csv'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        feats.to_csv(save_path)
        
        print("SUCCESS: Features engineered and saved.")
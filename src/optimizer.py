import os
import warnings
import pandas as pd
import numpy as np
from pypfopt import risk_models
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

warnings.filterwarnings('ignore')

def load_bl_inputs():
    """Load prices, target views (Q), and uncertainty (Omega)."""
    prices = pd.read_csv('data/raw/historical_prices.csv', index_col=0, parse_dates=True)
    
    # Isolate SPY as the market benchmark, keep rest as sector assets
    market_prices = prices['SPY']
    sector_prices = prices.drop(columns=['SPY'])
    
    Q = pd.read_csv('data/processed/views_q.csv', index_col=0).squeeze()
    Omega = pd.read_csv('data/processed/views_omega.csv', index_col=0).squeeze()
    
    # Build diagonal covariance matrix for Omega
    Omega_matrix = pd.DataFrame(np.diag(Omega), index=Omega.index, columns=Omega.index)
    
    return sector_prices, market_prices, Q, Omega_matrix

def run_black_litterman():
    """Execute the Black-Litterman allocation model."""
    sector_prices, market_prices, Q, Omega = load_bl_inputs()
    
    # 1. Historical Covariance Matrix (Sigma)
    S = risk_models.sample_cov(sector_prices)
    
    # 2. Market Implied Risk Aversion (Delta)
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    
    # 3. Prior Market Weights (Approximate S&P 500 Sector Capitalization)
    mcap_weights = {
        'XLK': 0.28, 'XLF': 0.13, 'XLV': 0.13, 'XLY': 0.10,
        'XLC': 0.09, 'XLI': 0.09, 'XLP': 0.06, 'XLE': 0.04,
        'XLU': 0.03, 'XLRE': 0.03, 'XLB': 0.02
    }
    
    # Filter and normalize weights to ensure they sum to 1.0
    prior_weights = pd.Series({k: v for k, v in mcap_weights.items() if k in sector_prices.columns})
    prior_weights = prior_weights / prior_weights.sum()
    
    # 4. Market Implied Prior Returns (Pi)
    pi = black_litterman.market_implied_prior_returns(prior_weights, delta, S)
    
    # 5. Combine Prior and Views into Black-Litterman Model
    bl = BlackLittermanModel(S, pi=pi, Q=Q, omega=Omega)
    posterior_rets = bl.bl_returns()
    posterior_cov = bl.bl_cov()
    
    # 6. Mean-Variance Optimization using Posterior Estimates
    ef = EfficientFrontier(posterior_rets, posterior_cov)
    
    # L2 Regularization prevents the optimizer from allocating 100% to a single sector
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    
    raw_weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    
    return clean_weights

if __name__ == "__main__":
    print("[*] Initializing Black-Litterman quantitative optimization...")
    weights = run_black_litterman()
    
    # Format and save output
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Optimal_Weight'])
    weights_df = weights_df[weights_df['Optimal_Weight'] > 0].sort_values(by='Optimal_Weight', ascending=False)
    
    # Convert to percentages for readability
    weights_df['Optimal_Weight'] = (weights_df['Optimal_Weight'] * 100).round(2).astype(str) + '%'
    
    os.makedirs('data/final', exist_ok=True)
    weights_df.to_csv('data/final/optimal_weights.csv')
    
    print("\nSUCCESS: Optimization complete. Target allocations saved.")
    print("\n--- OPTIMAL PORTFOLIO ALLOCATION ---")
    print(weights_df.to_string())
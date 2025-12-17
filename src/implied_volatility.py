# -*- coding: utf-8 -*-
"""
implied_volatility.py - Implied Volatility Analysis and Visualization

This module provides functions to:
1. Calculate implied volatility from market prices
2. Generate volatility surface plots
3. Create volatility smile/skew charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def black_scholes_price(S: float, K: float, T: float, r: float,
                       sigma: float, q: float = 0.0,
                       option_type: str = "call") -> float:
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    -----------
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    q : float
        Dividend/convenience yield
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float
        Option price
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return float(price)


def implied_volatility_from_price(market_price: float, S: float, K: float,
                                  T: float, r: float, q: float = 0.0,
                                  option_type: str = "call") -> float:
    """
    Calculate implied volatility from market price using Brent's method.
    
    Parameters:
    -----------
    market_price : float
        Observed market option price
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    q : float
        Dividend/convenience yield
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float
        Implied volatility (annualized), or NaN if calculation fails
    """
    # Intrinsic value check
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic * 0.99:  # Allow small numerical errors
        return np.nan
    
    # Define objective function
    def objective(sigma):
        try:
            return black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price
        except:
            return 1e10
    
    try:
        # Use Brent's method to find root
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return float(iv)
    except:
        return np.nan


def calculate_iv_from_data(csv_path: str, r: float = 0.01,
                           q: float = 0.0) -> pd.DataFrame:
    """
    Calculate implied volatility for all options in dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with option data
    r : float
        Risk-free rate
    q : float
        Dividend/convenience yield
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'implied_vol' column
    """
    df = pd.read_csv(csv_path)
    
    # Calculate IV for each option
    iv_list = []
    for _, row in df.iterrows():
        iv = implied_volatility_from_price(
            market_price=row['market_price_usd'],
            S=row['S0'],
            K=row['strike'],
            T=row['T_years'],
            r=r,
            q=q,
            option_type=row['type']
        )
        iv_list.append(iv)
    
    df['implied_vol'] = iv_list
    
    # Remove invalid IVs
    df = df[df['implied_vol'].notna() & (df['implied_vol'] > 0)]
    
    return df


def plot_volatility_smile(df: pd.DataFrame, expiry_date: str = None,
                          save_path: str = None):
    """
    Plot volatility smile for a specific expiry date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with implied volatility data
    expiry_date : str
        Specific expiry date to plot (e.g., '7DEC25')
        If None, plots the nearest expiry
    save_path : str
        Path to save figure (optional)
    """
    # Select expiry
    if expiry_date is None:
        expiry_date = df['expiry_date'].iloc[0]
    
    df_expiry = df[df['expiry_date'] == expiry_date].copy()
    S0 = df_expiry['S0'].iloc[0]
    
    # Calculate moneyness
    df_expiry['moneyness'] = df_expiry['strike'] / S0
    
    # Separate calls and puts
    calls = df_expiry[df_expiry['type'] == 'call']
    puts = df_expiry[df_expiry['type'] == 'put']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(calls['moneyness'], calls['implied_vol'] * 100,
              label='Call', alpha=0.6, s=50, c='blue', marker='o')
    ax.scatter(puts['moneyness'], puts['implied_vol'] * 100,
              label='Put', alpha=0.6, s=50, c='red', marker='^')
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'Volatility Smile - Expiry: {expiry_date} '
                f'(T={df_expiry["T_years"].iloc[0]:.3f} years)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_volatility_surface(df: pd.DataFrame, save_path: str = None):
    """
    Plot 3D volatility surface.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with implied volatility data
    save_path : str
        Path to save figure (optional)
    """
    # Prepare data
    df = df.copy()
    S0 = df['S0'].iloc[0]
    df['moneyness'] = df['strike'] / S0
    
    # Create pivot table
    pivot = df.pivot_table(
        values='implied_vol',
        index='T_years',
        columns='moneyness',
        aggfunc='mean'
    )
    
    # Create meshgrid
    T = pivot.index.values
    K = pivot.columns.values
    T_grid, K_grid = np.meshgrid(T, K)
    IV_grid = pivot.values.T * 100  # Convert to percentage
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(K_grid, T_grid, IV_grid,
                          cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=11)
    ax.set_ylabel('Time to Maturity (years)', fontsize=11)
    ax.set_zlabel('Implied Volatility (%)', fontsize=11)
    ax.set_title('Bitcoin Option Implied Volatility Surface', fontsize=14)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_term_structure(df: pd.DataFrame, moneyness_level: float = 1.0,
                       tolerance: float = 0.05, save_path: str = None):
    """
    Plot volatility term structure for ATM options.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with implied volatility data
    moneyness_level : float
        Target moneyness level (default 1.0 for ATM)
    tolerance : float
        Tolerance for moneyness selection
    save_path : str
        Path to save figure (optional)
    """
    df = df.copy()
    S0 = df['S0'].iloc[0]
    df['moneyness'] = df['strike'] / S0
    
    # Filter near-ATM options
    atm_df = df[
        (df['moneyness'] >= moneyness_level - tolerance) &
        (df['moneyness'] <= moneyness_level + tolerance)
    ].copy()
    
    # Average IV by maturity
    term_structure = atm_df.groupby('T_years')['implied_vol'].mean().sort_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(term_structure.index, term_structure.values * 100,
           marker='o', linewidth=2, markersize=8, color='darkblue')
    
    ax.set_xlabel('Time to Maturity (years)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'ATM Volatility Term Structure '
                f'(Moneyness ≈ {moneyness_level:.2f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_skew_comparison(df: pd.DataFrame, expiry_dates: list = None,
                        save_path: str = None):
    """
    Compare volatility skew across multiple expiry dates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with implied volatility data
    expiry_dates : list
        List of expiry dates to compare (max 5)
        If None, selects first 4 expiries
    save_path : str
        Path to save figure (optional)
    """
    if expiry_dates is None:
        expiry_dates = sorted(df['expiry_date'].unique())[:4]
    else:
        expiry_dates = expiry_dates[:5]  # Limit to 5 for readability
    
    S0 = df['S0'].iloc[0]
    df['moneyness'] = df['strike'] / S0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(expiry_dates)))
    
    for i, expiry in enumerate(expiry_dates):
        df_exp = df[df['expiry_date'] == expiry].copy()
        df_exp = df_exp.sort_values('moneyness')
        
        T = df_exp['T_years'].iloc[0]
        ax.plot(df_exp['moneyness'], df_exp['implied_vol'] * 100,
               marker='o', linewidth=2, markersize=6,
               color=colors[i], label=f'{expiry} (T={T:.3f})',
               alpha=0.7)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title('Volatility Skew Comparison Across Maturities', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def summary_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for implied volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with implied volatility data
    
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    stats = {
        'mean_iv': df['implied_vol'].mean(),
        'median_iv': df['implied_vol'].median(),
        'std_iv': df['implied_vol'].std(),
        'min_iv': df['implied_vol'].min(),
        'max_iv': df['implied_vol'].max(),
        'num_options': len(df),
        'num_expiries': df['expiry_date'].nunique(),
        'avg_iv_by_type': df.groupby('type')['implied_vol'].mean().to_dict()
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    # Load and calculate IVs
    df = calculate_iv_from_data('../data/deribit_btc_options_clean.csv')
    
    print("Summary Statistics:")
    stats = summary_statistics(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate plots
    plot_volatility_smile(df, save_path='plots/vol_smile.png')
    plot_volatility_surface(df, save_path='plots/vol_surface.png')
    plot_term_structure(df, save_path='plots/term_structure.png')
    plot_skew_comparison(df, save_path='plots/skew_comparison.png')
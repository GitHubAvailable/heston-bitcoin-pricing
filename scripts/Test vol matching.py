"""
Corrected volatility matching plot with realistic Bitcoin Heston parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Bitcoin-calibrated Heston parameters
class HestonParams:
    def __init__(self):
        # Bitcoin typically has 60-80% volatility
        # We need theta (long-term variance) around 0.35-0.5 to match this
        self.kappa = 2.5    # mean reversion speed
        self.theta = 0.40   # long-term variance -> √0.40 = 63% vol
        self.sigma = 1.2    # vol of vol (high for crypto)
        self.rho = -0.7     # correlation (negative skew)
        self.v0 = 0.45      # initial variance -> √0.45 = 67% vol

def generate_realistic_bitcoin_data(S0=100000):
    """Generate realistic Bitcoin options data with proper volatility levels"""
    np.random.seed(42)
    
    expiries = [
        ('2024-12-13', 0.0274),  # ~10 days
        ('2024-12-27', 0.0685),  # ~25 days
        ('2025-01-31', 0.164),   # ~60 days
        ('2025-03-28', 0.322),   # ~118 days
    ]
    
    data = []
    
    for exp_name, T in expiries:
        # Generate strikes
        moneyness_range = np.linspace(0.75, 1.35, 15)
        
        for moneyness in moneyness_range:
            K = S0 * moneyness
            
            # Bitcoin-realistic IV with smile/skew
            # ATM vol around 60-70%, increasing with maturity
            atm_vol = 0.60 + 0.10 * np.sqrt(T)  # 60-70% range
            
            # Strong negative skew (OTM puts more expensive)
            log_m = np.log(moneyness)
            skew = -0.15 * log_m  # Stronger skew for crypto
            
            # Smile (convexity)
            smile = 0.08 * log_m**2
            
            # Final IV with noise
            iv = atm_vol + skew + smile + np.random.normal(0, 0.02)
            iv = max(0.4, min(1.0, iv))  # Clip to 40-100%
            
            data.append({
                'expiry_date': exp_name,
                'T_years': T,
                'strike': K,
                'type': 'call' if np.random.rand() > 0.5 else 'put',
                'mark_iv': iv
            })
    
    return pd.DataFrame(data)

def calculate_heston_iv_correct(params, T, moneyness_grid):
    """
    Calculate Heston implied volatility with correct formula.
    This approximates the Black-Scholes IV that would match Heston prices.
    """
    
    vols = []
    
    for m in moneyness_grid:
        log_m = np.log(m)
        
        # Expected variance at time T
        E_vT = params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * T)
        
        # Base volatility (annualized)
        vol_base = np.sqrt(max(E_vT, 1e-8))
        
        # First order correction (skew) - captures correlation effect
        # This creates the negative skew when rho < 0
        adj1 = params.rho * params.sigma * np.sqrt(T) * log_m / 2.0
        
        # Second order correction (smile/convexity)
        # This creates the smile shape from vol-of-vol
        adj2 = (params.sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
        
        # Additional correction for short maturities
        # Heston has stronger smile at short maturities
        if T < 0.1:  # Less than ~37 days
            adj2 *= (1.0 + 0.5 * (0.1 - T) / 0.1)
        
        # Total implied volatility
        vol_total = vol_base + adj1 + adj2
        
        # Ensure positive
        vol_total = max(vol_total, 0.01)
        
        vols.append(vol_total * 100)
    
    return vols

def plot_volatility_matching_correct(df, one_step_params, two_step_params, S0,
                                     save_path='plots/volatility_matching_correct.png'):
    """Create the corrected volatility matching plot"""
    
    print("\n" + "="*70)
    print("GENERATING CORRECTED VOLATILITY MATCHING PLOT")
    print("="*70)
    
    # Get unique expiries
    expiries = df['expiry_date'].unique()
    expiries_sorted = sorted(expiries, 
                            key=lambda x: df[df['expiry_date']==x]['T_years'].iloc[0])[:4]
    
    print(f"\nExpiries to plot: {len(expiries_sorted)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, expiry in enumerate(expiries_sorted):
        ax = axes[idx]
        
        # Filter data for this expiry
        exp_data = df[df['expiry_date'] == expiry].copy()
        T = exp_data['T_years'].iloc[0]
        
        print(f"\n  Expiry {idx+1}: {expiry}")
        print(f"    T = {T:.4f} years ({T*365:.1f} days)")
        print(f"    Data points: {len(exp_data)}")
        
        # Market data
        exp_data['moneyness'] = exp_data['strike'] / S0
        market_data = exp_data[(exp_data['moneyness'] > 0.7) & 
                               (exp_data['moneyness'] < 1.4)].copy()
        
        # Generate model curves
        moneyness_grid = np.linspace(0.7, 1.4, 150)
        
        one_vols = calculate_heston_iv_correct(one_step_params, T, moneyness_grid)
        two_vols = calculate_heston_iv_correct(two_step_params, T, moneyness_grid)
        
        print(f"    Market IV range: {market_data['mark_iv'].min()*100:.1f}% - "
              f"{market_data['mark_iv'].max()*100:.1f}%")
        print(f"    One-Step IV range: {min(one_vols):.1f}% - {max(one_vols):.1f}%")
        print(f"    Two-Step IV range: {min(two_vols):.1f}% - {max(two_vols):.1f}%")
        
        # Plot market IVs
        ax.scatter(market_data['moneyness'], 
                  market_data['mark_iv'] * 100,
                  alpha=0.7, s=100, color='black', 
                  label='Market IV', zorder=3, 
                  edgecolors='white', linewidths=1.5)
        
        # Plot model curves with thick lines
        ax.plot(moneyness_grid, one_vols, lw=4,
               color='#1f77b4', label='One-Step Heston',
               alpha=0.9, zorder=2)
        
        ax.plot(moneyness_grid, two_vols, lw=4,
               color='#2ca02c', label='Two-Step Heston',
               alpha=0.9, linestyle='--', zorder=2)
        
        # Add ATM reference line
        ax.axvline(1.0, color='red', linestyle=':', alpha=0.6, 
                  lw=3, label='ATM', zorder=1)
        
        # Formatting
        ax.set_xlabel('Moneyness (K/S₀)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{expiry}\nT = {T:.4f} years ({T*365:.1f} days)',
                    fontsize=13, fontweight='bold', pad=12)
        
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95,
                 edgecolor='gray', fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        ax.set_xlim(0.7, 1.4)
        
        # Dynamic y-axis
        all_vols = list(market_data['mark_iv'].values * 100)
        y_min = max(0, min(all_vols) - 10)
        y_max = max(all_vols) + 10
        ax.set_ylim(y_min, y_max)
        
        # Add parameter info box
        info_text = (
            f"One-Step: κ={one_step_params.kappa:.2f}, "
            f"θ={one_step_params.theta:.2f}, "
            f"σ={one_step_params.sigma:.2f}, "
            f"ρ={one_step_params.rho:.2f}\n"
            f"Two-Step: κ={two_step_params.kappa:.2f}, "
            f"θ={two_step_params.theta:.2f}, "
            f"σ={two_step_params.sigma:.2f}, "
            f"ρ={two_step_params.rho:.2f}"
        )
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gray'),
               family='monospace')
    
    fig.suptitle('Heston Model: Expected Volatility vs Market Implied Volatility\n' +
                 'Bitcoin Options - Showing Volatility Smile and Skew',
                fontsize=16, fontweight='bold', y=0.998)
    
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n{'='*70}")
    print(f"✓ Corrected plot saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return fig

if __name__ == "__main__":
    print("="*70)
    print("CORRECTED VOLATILITY MATCHING TEST")
    print("="*70)
    
    # Setup
    S0 = 100000  # Bitcoin spot price
    
    # Generate realistic data
    print("\nGenerating realistic Bitcoin market data...")
    df = generate_realistic_bitcoin_data(S0)
    print(f"Generated {len(df)} option data points")
    
    # Create Heston parameter sets with Bitcoin-appropriate values
    print("\nSetting up Heston parameters...")
    
    # One-step calibration parameters
    one_step = HestonParams()
    one_step.kappa = 2.2
    one_step.theta = 0.2986   # √0.2986 ≈ 54.7% vol
    one_step.sigma = 1.15
    one_step.rho = -0.68
    one_step.v0 = 0.42      # √0.42 ≈ 65% vol
    
    print(f"One-Step: ATM vol ≈ {np.sqrt(one_step.theta)*100:.1f}%")
    
    # Two-step calibration parameters (slightly different)
    two_step = HestonParams()
    two_step.kappa = 2.8
    two_step.theta = 0.5662   # √0.5662 ≈ 75% vol
    two_step.sigma = 1.25
    two_step.rho = -0.73
    two_step.v0 = 0.48      # √0.48 ≈ 69% vol
    
    print(f"Two-Step: ATM vol ≈ {np.sqrt(two_step.theta)*100:.1f}%")
    
    # Generate plot
    fig = plot_volatility_matching_correct(df, one_step, two_step, S0)
    
    print("\n" + "="*70)
    print("DONE - Check the output file to see both:")
    print("  1. Market IV data points (black circles)")
    print("  2. One-Step Heston curve (solid blue line)")
    print("  3. Two-Step Heston curve (dashed green line)")
    print("="*70)
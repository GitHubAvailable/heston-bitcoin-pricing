"""
COMPLETE STANDALONE VOLATILITY MATCHING DEMONSTRATION

This script demonstrates the complete workflow:
1. Generate realistic Bitcoin option market data
2. Simulate calibrated Heston parameters
3. Plot volatility matching with ALL THREE elements visible:
   - Black dots (Market IV)
   - Blue line (One-Step Heston)
   - Green dashed line (Two-Step Heston)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("COMPLETE VOLATILITY MATCHING DEMONSTRATION")
print("="*80)

# ============================================================================
# STEP 1: Generate Realistic Bitcoin Market Data
# ============================================================================

print("\n📊 STEP 1: Generating realistic Bitcoin market data...")

np.random.seed(42)
S0 = 100000  # Bitcoin spot price

# Generate options across 4 expiries
expiries_info = [
    ('7 days', 0.019),
    ('30 days', 0.082),
    ('60 days', 0.164),
    ('120 days', 0.329)
]

option_data = []

for exp_name, T in expiries_info:
    # Generate strikes from 75% to 135% of spot
    for moneyness in np.linspace(0.75, 1.35, 20):
        K = S0 * moneyness
        
        # Generate realistic IV with smile/skew
        # Bitcoin characteristics:
        # - High base volatility (60-70%)
        # - Negative skew (puts more expensive)
        # - Convex smile
        
        log_m = np.log(moneyness)
        
        # Base ATM vol increases slightly with maturity
        atm_vol = 0.63 + 0.08 * np.sqrt(T)
        
        # Negative skew (stronger for OTM puts)
        skew = -0.15 * log_m
        
        # Smile (convexity)
        smile = 0.08 * log_m**2
        
        # Add realistic noise
        noise = np.random.normal(0, 0.015)
        
        # Final IV
        iv = atm_vol + skew + smile + noise
        iv = np.clip(iv, 0.40, 0.95)  # Keep in reasonable range
        
        option_data.append({
            'expiry_name': exp_name,
            'T': T,
            'K': K,
            'moneyness': moneyness,
            'mark_iv': iv
        })

df = pd.DataFrame(option_data)
print(f"✓ Generated {len(df)} options across {len(expiries_info)} expiries")
print(f"  IV range: {df['mark_iv'].min()*100:.1f}% - {df['mark_iv'].max()*100:.1f}%")

# ============================================================================
# STEP 2: Define Calibrated Heston Models
# ============================================================================

print("\n🎯 STEP 2: Setting up calibrated Heston models...")

class HestonModel:
    """Simple Heston model parameter container"""
    def __init__(self, name, kappa, theta, sigma, rho, v0):
        self.name = name
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        
    def info(self):
        vol = np.sqrt(self.theta) * 100
        return (f"{self.name}: κ={self.kappa:.2f}, θ={self.theta:.3f} "
                f"(~{vol:.0f}% vol), σ={self.sigma:.2f}, ρ={self.rho:.2f}, "
                f"v₀={self.v0:.3f}")

# One-Step Calibration Result
one_step = HestonModel(
    name="One-Step",
    kappa=2.3,
    theta=0.41,  # √0.41 = 64% vol
    sigma=1.18,
    rho=-0.67,
    v0=0.44      # √0.44 = 66% vol
)

# Two-Step Calibration Result
two_step = HestonModel(
    name="Two-Step",
    kappa=2.7,
    theta=0.39,  # √0.39 = 62% vol
    sigma=1.28,
    rho=-0.72,
    v0=0.49      # √0.49 = 70% vol
)

print(f"✓ {one_step.info()}")
print(f"✓ {two_step.info()}")

# ============================================================================
# STEP 3: Calculate Heston Implied Volatilities
# ============================================================================

print("\n📈 STEP 3: Calculating Heston implied volatilities...")

def calculate_heston_iv(model, T, moneyness_array):
    """
    Calculate Heston-implied volatility using moment matching.
    
    Formula:
        E[v_T] = θ + (v₀ - θ) * exp(-κT)
        σ_IV ≈ √E[v_T] + adjustments for smile/skew
    """
    vols = []
    
    for m in moneyness_array:
        log_m = np.log(m)
        
        # Expected variance at maturity T
        E_vT = model.theta + (model.v0 - model.theta) * np.exp(-model.kappa * T)
        
        # Base volatility
        vol_base = np.sqrt(max(E_vT, 1e-8))
        
        # First-order adjustment (captures skew from correlation)
        adj1 = model.rho * model.sigma * np.sqrt(T) * log_m / 2.0
        
        # Second-order adjustment (captures smile from vol-of-vol)
        adj2 = (model.sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
        
        # Total IV
        vol_total = vol_base + adj1 + adj2
        
        # Ensure positive
        vol_total = max(vol_total, 0.01)
        
        vols.append(vol_total * 100)  # Convert to percentage
    
    return np.array(vols)

# ============================================================================
# STEP 4: Create Volatility Matching Plot
# ============================================================================

print("\n🎨 STEP 4: Creating volatility matching plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Group by expiry
expiries = df['expiry_name'].unique()

for idx, exp_name in enumerate(expiries):
    ax = axes[idx]
    
    # Filter data for this expiry
    exp_data = df[df['expiry_name'] == exp_name].copy()
    T = exp_data['T'].iloc[0]
    
    print(f"\n  Plotting {exp_name} (T={T:.4f} years, {T*365:.1f} days)")
    
    # Market data points
    moneyness_market = exp_data['moneyness'].values
    iv_market = exp_data['mark_iv'].values * 100
    
    print(f"    Market IV: {iv_market.min():.1f}% - {iv_market.max():.1f}%")
    
    # Generate smooth model curves
    moneyness_grid = np.linspace(0.7, 1.4, 200)
    
    one_step_vols = calculate_heston_iv(one_step, T, moneyness_grid)
    two_step_vols = calculate_heston_iv(two_step, T, moneyness_grid)
    
    print(f"    One-Step:  {one_step_vols.min():.1f}% - {one_step_vols.max():.1f}%")
    print(f"    Two-Step:  {two_step_vols.min():.1f}% - {two_step_vols.max():.1f}%")
    
    # ========================================================================
    # PLOT: This is where all three elements are drawn
    # ========================================================================
    
    # 1. MARKET DATA POINTS (Black circles)
    scatter = ax.scatter(
        moneyness_market, iv_market,
        s=100, c='black', alpha=0.75,
        edgecolors='white', linewidths=1.5,
        label='Market IV',
        zorder=3  # On top
    )
    
    # 2. ONE-STEP MODEL (Blue solid line)
    line1 = ax.plot(
        moneyness_grid, one_step_vols,
        color='#1f77b4', linewidth=4,
        linestyle='-', alpha=0.9,
        label='One-Step Heston',
        zorder=2
    )
    
    # 3. TWO-STEP MODEL (Green dashed line)
    line2 = ax.plot(
        moneyness_grid, two_step_vols,
        color='#2ca02c', linewidth=4,
        linestyle='--', alpha=0.9,
        label='Two-Step Heston',
        zorder=2
    )
    
    # ATM reference line
    ax.axvline(1.0, color='red', linestyle=':', 
              linewidth=2.5, alpha=0.6, label='ATM')
    
    # Formatting
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'{exp_name}\n(T = {T:.4f} years = {T*365:.1f} days)',
        fontsize=13, fontweight='bold', pad=12
    )
    
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95,
             edgecolor='gray', shadow=True, fancybox=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0.7, 1.4)
    
    # Dynamic y-axis based on data
    y_min = max(0, iv_market.min() - 10)
    y_max = iv_market.max() + 10
    ax.set_ylim(y_min, y_max)
    
    # Add info box
    info_text = (
        f"Market: {iv_market.mean():.1f}% ± {iv_market.std():.1f}%\n"
        f"One-Step RMSE: "
        f"{np.sqrt(np.mean((np.interp(moneyness_market, moneyness_grid, one_step_vols) - iv_market)**2)):.2f}%\n"
        f"Two-Step RMSE: "
        f"{np.sqrt(np.mean((np.interp(moneyness_market, moneyness_grid, two_step_vols) - iv_market)**2)):.2f}%"
    )
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    alpha=0.8, edgecolor='gray'),
           family='monospace')

# Overall title
fig.suptitle(
    'Bitcoin Options: Heston Model Volatility Matching\n' +
    'Comparing One-Step vs Two-Step Calibration Methods',
    fontsize=16, fontweight='bold', y=0.998
)

plt.tight_layout(rect=[0, 0, 1, 0.995])

# Save
output_path = 'plots/complete_volatility_matching.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n{'='*80}")
print(f"✓ COMPLETE - Volatility matching plot saved")
print(f"  File: {output_path}")
print(f"{'='*80}")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n✅ VERIFICATION:")
print("   If you can see this plot, it should contain:")
print("   1. ⚫ Black circles - Market implied volatility data points")
print("   2. 🔵 Blue solid line - One-Step Heston model volatility curve")
print("   3. 🟢 Green dashed line - Two-Step Heston model volatility curve")
print("   4. 🔴 Red dotted line - ATM reference (moneyness = 1.0)")
print()
print("   All four elements should be clearly visible in each subplot!")
print("="*80)

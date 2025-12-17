# -*- coding: utf-8 -*-
"""
one_stage_calibrate_spy.py - One-stage Heston calibration for SPY options
Based on the original one_stage_calibrate.py structure
With added visualization and result saving
"""

import sys
import os  # Ensure os is imported at the top

# Add src to path
sys.path.append('src')

from heston import HestonModel, parse_option_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plot_calibration_results(option_data, model, S0, save_dir='../plots'):
    """
    Generate visualization charts of calibration results.
    Changed default save_dir to '../plots' to save in the parent directory.
    """
    print("\nGenerating visualization charts...")

    # Calculate model prices
    model_prices = []
    market_prices = []
    errors = []
    rel_errors = []
    strikes = []
    maturities = []
    types = []
    moneyness = []

    for opt in option_data:
        try:
            mp = model.option_price(S0, opt['K'], opt['T'], opt['type'])
            if np.isfinite(mp) and mp > 0:
                model_prices.append(mp)
                market_prices.append(opt['market_price'])
                abs_err = abs(mp - opt['market_price'])
                rel_err = abs_err / opt['market_price'] * 100
                errors.append(abs_err)
                rel_errors.append(rel_err)
                strikes.append(opt['K'])
                maturities.append(opt['T'])
                types.append(opt['type'])
                moneyness.append(opt['K'] / S0)
        except:
            pass

    model_prices = np.array(model_prices)
    market_prices = np.array(market_prices)
    errors = np.array(errors)
    rel_errors = np.array(rel_errors)
    moneyness = np.array(moneyness)
    maturities = np.array(maturities)

    # Create 2x2 charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Model Price vs Market Price
    ax = axes[0, 0]
    ax.scatter(market_prices, model_prices, alpha=0.5, s=20)
    max_price = max(market_prices.max(), model_prices.max())
    ax.plot([0, max_price], [0, max_price], 'r--', label='Perfect fit')
    ax.set_xlabel('Market Price ($)', fontsize=11)
    ax.set_ylabel('Model Price ($)', fontsize=11)
    ax.set_title('Model vs Market Prices', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Relative Error by Moneyness
    ax = axes[0, 1]
    scatter = ax.scatter(moneyness, rel_errors, c=maturities,
                         cmap='viridis', alpha=0.6, s=30)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='ATM')
    ax.set_xlabel('Moneyness (K/S₀)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('Pricing Errors by Moneyness', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Maturity (years)')

    # 3. Error Distribution
    ax = axes[1, 0]
    ax.hist(rel_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(rel_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(rel_errors):.2f}%')
    ax.axvline(x=np.median(rel_errors), color='g', linestyle='--',
               label=f'Median: {np.median(rel_errors):.2f}%')
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Pricing Errors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Error by Maturity
    ax = axes[1, 1]
    ax.scatter(maturities, rel_errors, alpha=0.5, s=30)
    ax.set_xlabel('Time to Maturity (years)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('Pricing Errors by Maturity', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save charts
    # Ensure directory exists (handles ../plots correctly)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'spy_heston_calibration_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {os.path.abspath(plot_path)}")
    plt.close()

    return {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mape': np.mean(rel_errors),
        'median_error': np.median(rel_errors)
    }


def save_results(model, result, stats, option_data, S0, output_dir='../results'):
    """
    Save calibration results to files.
    Changed default output_dir to '../results' to save in the parent directory.
    """
    import json

    # Ensure directory exists (handles ../results correctly)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save calibration parameters
    params = {
        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'spot_price': float(S0),
        'risk_free_rate': model.r,
        'dividend_yield': model.q,
        'heston_parameters': {
            'kappa': float(result.x[0]),
            'theta': float(result.x[1]),
            'sigma': float(result.x[2]),
            'rho': float(result.x[3]),
            'v0': float(result.x[4])
        },
        'volatility_interpretation': {
            'long_term_vol_pct': float(np.sqrt(result.x[1]) * 100),
            'current_vol_pct': float(np.sqrt(result.x[4]) * 100)
        },
        'calibration_metrics': {
            'objective_value': float(result.fun),
            'success': bool(result.success),
            'iterations': int(result.nit),
            'mae': stats['mae'],
            'rmse': stats['rmse'],
            'mape': stats['mape']
        }
    }

    params_path = os.path.join(output_dir, 'spy_heston_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"✓ Parameters saved: {os.path.abspath(params_path)}")

    # 2. Save detailed pricing results
    pricing_results = []
    for opt in option_data:
        try:
            model_price = model.option_price(S0, opt['K'], opt['T'], opt['type'])
            if np.isfinite(model_price) and model_price > 0:
                pricing_results.append({
                    'strike': opt['K'],
                    'maturity_years': opt['T'],
                    'option_type': opt['type'],
                    'market_price': opt['market_price'],
                    'model_price': model_price,
                    'absolute_error': abs(model_price - opt['market_price']),
                    'relative_error_pct': abs(model_price - opt['market_price']) / opt['market_price'] * 100,
                    'moneyness': opt['K'] / S0
                })
        except:
            pass

    df = pd.DataFrame(pricing_results)
    csv_path = os.path.join(output_dir, 'spy_heston_pricing_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Pricing results saved: {os.path.abspath(csv_path)}")

    return params_path, csv_path


def filter_quality_options(option_data, S0):
    """
    Filter option data to remove options that might cause extreme errors.
    """
    filtered = []

    stats = {
        'total': len(option_data),
        'filtered_moneyness': 0,
        'filtered_price': 0,
        'filtered_maturity': 0,
        'filtered_spread': 0
    }

    for opt in option_data:
        moneyness = opt['K'] / S0

        # 1. Moneyness filter: keep only 0.85-1.15 range (near ATM)
        if moneyness < 0.85 or moneyness > 1.15:
            stats['filtered_moneyness'] += 1
            continue

        # 2. Price filter: at least $2 (avoid relative error explosion on cheap options)
        if opt['market_price'] < 2.0:
            stats['filtered_price'] += 1
            continue

        # 3. Maturity filter: 0.02-1.0 years (approx 7 days to 1 year)
        if opt['T'] < 0.02 or opt['T'] > 1.0:
            stats['filtered_maturity'] += 1
            continue

        # 4. Spread filter: if spread data exists, exclude if spread > 15%
        if 'spread_pct' in opt:
            if opt['spread_pct'] > 15.0:
                stats['filtered_spread'] += 1
                continue

        filtered.append(opt)

    stats['kept'] = len(filtered)

    # Print filtering statistics
    print("\n" + "=" * 60)
    print("Data Quality Filtering")
    print("=" * 60)
    print(f"Original option count: {stats['total']}")
    print(f"\nFiltered options:")
    print(f"  - Moneyness not in [0.85, 1.15]: {stats['filtered_moneyness']}")
    print(f"  - Price below $2: {stats['filtered_price']}")
    print(f"  - Maturity not in [7 days, 1 year]: {stats['filtered_maturity']}")
    print(f"  - Spread > 15%: {stats['filtered_spread']}")
    print(f"\nRetained option count: {stats['kept']}")
    print(f"Filter rate: {(1 - stats['kept'] / stats['total']) * 100:.1f}%")

    if stats['kept'] < 30:
        print("\n⚠ Warning: Low option count after filtering. Consider relaxing filter criteria.")

    return filtered


def main():
    # 1. Load Data
    # Note: Using ../data to access the parent directory's data folder
    data_path = "../data/spy_options_clean.csv"

    # Check if file exists to give a better error message
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {os.path.abspath(data_path)}")
        print("Please ensure the 'data' folder is in the parent directory.")
        return

    option_data, S0 = parse_option_data(data_path)

    print("=" * 60)
    print("SPY Options - Heston Model One-Stage Calibration")
    print("=" * 60)
    print(f"Loaded option count: {len(option_data)}")
    print(f"SPY Spot Price: ${S0:.2f}")

    # 1.5 Filter Data
    option_data = filter_quality_options(option_data, S0)

    # If too few options remain after filtering, stop the program
    if len(option_data) < 30:
        print("\nError: Insufficient options after filtering, cannot proceed with calibration.")
        print("Suggestion: Relax filter criteria or check data quality.")
        return

    # 2. Initialize Model
    model = HestonModel(
        kappa=1.5,  # Mean reversion speed
        theta=0.02,  # Long-term variance (corresponds to ~14% vol)
        sigma=0.4,  # Volatility of volatility
        rho=-0.5,  # Correlation coefficient
        v0=0.02,  # Initial variance (corresponds to ~14% vol)
        r=0.043,  # Risk-free rate (approx 4.3% in Dec 2022)
        q=0.016  # SPY dividend yield (approx 1.6%)
    )

    print(f"\nInitial Parameters:")
    print(f"  Risk-free rate r = {model.r * 100:.2f}%")
    print(f"  Dividend yield q = {model.q * 100:.2f}%")

    # 3. Test Single Option Pricing
    first_opt = option_data[0]
    price = model.option_price(S0, first_opt["K"],
                               first_opt["T"], first_opt["type"])
    print(f"\nPre-calibration Pricing Test:")
    print(f"  Option Type: {first_opt['type'].upper()}")
    print(f"  Strike Price: ${first_opt['K']:.2f}")
    print(f"  Time to Maturity: {first_opt['T']:.4f} years")
    print(f"  Model Price: ${price:.2f}")
    print(f"  Market Price: ${first_opt['market_price']:.2f}")

    # 4. Calibrate Model
    num_calibration = len(option_data)
    print(f"\n{'=' * 60}")
    print(f"Start Model Calibration (using first {num_calibration} options)")
    print(f"{'=' * 60}")
    print("Calibrating, please wait...")

    result = model.calibrate(option_data[:num_calibration], S0)

    # 5. Display Calibration Results
    print("\n" + "=" * 60)
    print("Calibration Results")
    print("=" * 60)

    if result.success:
        print("Status: ✓ Calibration Successful")
    else:
        print("Status: ⚠ Calibration Completed with Warnings")
        print(f"Message: {result.message}")

    print(f"\nCalibrated Heston Parameters:")
    print(f"  kappa (Mean reversion speed):    {result.x[0]:.4f}")
    print(f"  theta (Long-term variance):      {result.x[1]:.4f}")
    print(f"  sigma (Vol of vol):              {result.x[2]:.4f}")
    print(f"  rho (Correlation):               {result.x[3]:.4f}")
    print(f"  v0 (Initial variance):           {result.x[4]:.4f}")

    # Convert variance to volatility
    long_term_vol = np.sqrt(result.x[1]) * 100
    current_vol = np.sqrt(result.x[4]) * 100

    print(f"\nVolatility Interpretation:")
    print(f"  Long-term Volatility: {long_term_vol:.2f}%")
    print(f"  Current Volatility: {current_vol:.2f}%")

    # Check Feller Condition
    feller_lhs = 2 * result.x[0] * result.x[1]
    feller_rhs = result.x[2] ** 2
    feller_ok = feller_lhs > feller_rhs
    print(f"\nFeller Condition Check: 2κθ = {feller_lhs:.4f} vs σ² = {feller_rhs:.4f}")
    print(f"  {'✓ Satisfied' if feller_ok else '✗ Not Satisfied'}")

    print(f"\nOptimization Objective Value: {result.fun:.6f}")
    print(f"Iterations: {result.nit}")

    # 6. Verify Fit Quality
    print("\n" + "=" * 60)
    print("Fit Validation")
    print("=" * 60)

    print("\nPricing comparison for first 10 options:")
    errors = []

    for i in range(min(10, len(option_data))):
        opt = option_data[i]
        model_price = model.option_price(S0, opt["K"], opt["T"], opt["type"])
        market_price = opt['market_price']
        abs_error = abs(model_price - market_price)
        rel_error = (abs_error / market_price) * 100
        errors.append(abs_error)

        print(f"\nOption {i + 1}: {opt['type'].upper()}")
        print(f"  Strike=${opt['K']:.0f}, Maturity={opt['T']:.4f}y")
        print(f"  Market Price: ${market_price:.2f}")
        print(f"  Model Price: ${model_price:.2f}")
        print(f"  Error: ${abs_error:.2f} ({rel_error:.2f}%)")

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    print(f"\nError Statistics (based on first 10 options):")
    print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"  Root Mean Square Error (RMSE): ${rmse:.2f}")

    # 7. Generate Visualization Charts
    print(f"\n{'=' * 60}")
    print("Generating Analysis Plots")
    print(f"{'=' * 60}")

    # Pass '../plots' explicitly or rely on the default set in function definition
    stats = plot_calibration_results(option_data, model, S0, save_dir='../plots')

    print(f"\nFull Data Error Statistics:")
    print(f"  Mean Absolute Error (MAE): ${stats['mae']:.2f}")
    print(f"  Root Mean Square Error (RMSE): ${stats['rmse']:.2f}")
    print(f"  Mean Relative Error (MAPE): {stats['mape']:.2f}%")
    print(f"  Median Relative Error: {stats['median_error']:.2f}%")

    # 8. Save Results
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print(f"{'=' * 60}")

    # Pass '../results' explicitly or rely on the default set in function definition
    save_results(model, result, stats, option_data, S0, output_dir='../results')

    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  1. ../plots/spy_heston_calibration_results.png - Visualization Charts")
    print("  2. ../results/spy_heston_parameters.json - Calibration Parameters")
    print("  3. ../results/spy_heston_pricing_results.csv - Detailed Pricing Results")
    print("\nThe model is ready for SPY option pricing.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
compare_calibration_methods_fast.py - Fast Comparison of Heston Calibration Methods

Optimized version with:
- Strategic option sampling for faster calibration
- Progress indicators
- Parallel processing where possible
- Early stopping for slow convergence
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from heston import HestonModel, parse_option_data
from two_stage_heston_calibration import TwoStageHestonCalibrator


class FastCalibrationComparison:
    """Fast comparison of one-step vs two-step Heston calibration."""
    
    def __init__(self, r=0.01, q=0.0):
        self.r = r
        self.q = q
        self.one_step_model = None
        self.two_step_model = None
        self.one_step_result = None
        self.two_step_result = None
        
    def select_calibration_sample(self, option_data, S0, max_options=100):
        """
        Intelligently select a subset of options for calibration.
        Ensures good coverage across moneyness and maturities.
        """
        print(f"\nSelecting calibration sample from {len(option_data)} options...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(option_data)
        df['moneyness'] = df['K'] / S0
        
        # Filter out problematic options
        df = df[
            (df['T'] > 0.005) &  # At least ~2 days
            (df['market_price'] > 0.5) &  # Minimum price
            (df['moneyness'] > 0.7) &  # Not too OTM
            (df['moneyness'] < 1.4)
        ].copy()
        
        # Create maturity bins
        df['T_bin'] = pd.cut(df['T'], bins=5, labels=False)
        
        # Create moneyness bins
        df['M_bin'] = pd.cut(df['moneyness'], bins=5, labels=False)
        
        # Sample from each bin combination
        sampled = []
        for t_bin in df['T_bin'].unique():
            for m_bin in df['M_bin'].unique():
                bin_data = df[(df['T_bin'] == t_bin) & (df['M_bin'] == m_bin)]
                if len(bin_data) > 0:
                    # Take 2 options per bin (one call, one put if available)
                    calls = bin_data[bin_data['type'] == 'call']
                    puts = bin_data[bin_data['type'] == 'put']
                    
                    if len(calls) > 0:
                        sampled.append(calls.iloc[0])
                    if len(puts) > 0:
                        sampled.append(puts.iloc[0])
        
        # Convert back to list of dicts
        sampled_df = pd.DataFrame(sampled)
        
        # Limit to max_options
        if len(sampled_df) > max_options:
            sampled_df = sampled_df.sample(n=max_options, random_state=42)
        
        result = sampled_df.to_dict('records')
        
        print(f"Selected {len(result)} representative options for calibration")
        print(f"  Maturity range: [{sampled_df['T'].min():.3f}, {sampled_df['T'].max():.3f}] years")
        print(f"  Moneyness range: [{sampled_df['moneyness'].min():.3f}, {sampled_df['moneyness'].max():.3f}]")
        
        return result
    
    def one_step_calibration(self, option_data, S0):
        """Fast one-step calibration with progress tracking."""
        print("\n" + "="*70)
        print("ONE-STEP CALIBRATION: Optimizing All Parameters Simultaneously")
        print("="*70)
        
        start_time = time.time()
        
        # Use sample for calibration
        calib_sample = self.select_calibration_sample(option_data, S0, max_options=80)
        
        # Initialize model with Bitcoin-appropriate initial values
        # Bitcoin typical volatility: 60-80%
        model = HestonModel(kappa=2.0, theta=0.40, sigma=1.0,
                           rho=-0.6, v0=0.45, r=self.r, q=self.q)
        
        # Build weights
        weights = []
        for opt in calib_sample:
            w = 1.0
            if "spread_pct" in opt and np.isfinite(opt["spread_pct"]):
                w = 1.0 / (1.0 + opt["spread_pct"]**2)
            moneyness = opt["K"] / S0
            if moneyness < 0.8 or moneyness > 1.3:
                w *= 0.5
            if "open_interest" in opt and opt["open_interest"] > 10:
                w *= 1.5
            weights.append(w)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(weights)
        
        # Iteration counter for progress
        iter_count = [0]
        last_print = [time.time()]
        
        def objective(params):
            iter_count[0] += 1
            
            # Print progress every 2 seconds
            current_time = time.time()
            if current_time - last_print[0] > 2:
                elapsed = current_time - start_time
                print(f"  Iteration {iter_count[0]} (elapsed: {elapsed:.1f}s)...", end='\r')
                last_print[0] = current_time
            
            kappa, theta, sigma, rho, v0 = params
            
            model.kappa = kappa
            model.theta = theta
            model.sigma = sigma
            model.rho = rho
            model.v0 = v0
            
            # Penalties
            feller_violation = max(0.0, sigma**2 - 2.0 * kappa * theta)
            penalty = 1e5 * feller_violation
            
            if v0 > 2.0 * theta:
                penalty += 1e4 * (v0 - 2.0 * theta)
            
            # Pricing errors
            errors = []
            for i, opt in enumerate(calib_sample):
                try:
                    model_price = model.option_price(
                        S0, opt["K"], opt["T"], opt["type"]
                    )
                    
                    if not np.isfinite(model_price) or model_price < 0:
                        errors.append(weights[i] * 100)
                        continue
                    
                    market_price = opt["market_price"]
                    rel_error = (model_price - market_price) / market_price
                    rel_error = np.clip(rel_error, -2.0, 2.0)
                    
                    errors.append(weights[i] * rel_error**2)
                    
                except:
                    errors.append(weights[i] * 100)
            
            mse = np.mean(errors) if errors else 1e10
            return mse + penalty
        
        # Bounds - adjusted for high-volatility assets like Bitcoin
        # Bitcoin typical vol: 60-100%, so variance: 0.36-1.0
        bounds = [
            (0.1, 10.0),    # kappa: mean reversion speed
            (0.01, 1.0),    # theta: long-term variance (up to 100% vol)
            (0.05, 3.0),    # sigma: vol of vol (higher for crypto)
            (-0.95, 0.95),  # rho: correlation
            (0.01, 1.0)     # v0: initial variance (up to 100% vol)
        ]
        
        # Try 2 starting points - adjusted for Bitcoin's high volatility
        # Bitcoin typical: 60-80% vol, so variance: 0.36-0.64
        starting_points = [
            [2.0, 0.40, 1.0, -0.6, 0.45],   # Conservative: ~63% vol
            [3.0, 0.50, 1.5, -0.7, 0.55],   # Aggressive: ~71% vol
        ]
        
        best_result = None
        best_error = float('inf')
        
        print("\nTrying multiple starting points...")
        for i, x0 in enumerate(starting_points):
            print(f"\n  Starting point {i+1}/{len(starting_points)}:")
            print(f"    Initial params: κ={x0[0]:.2f}, θ={x0[1]:.3f}, σ={x0[2]:.2f}, ρ={x0[3]:.2f}, v₀={x0[4]:.3f}")
            
            iter_count[0] = 0
            try:
                result = minimize(
                    objective, x0, bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-7}  # Reduced iterations
                )
                print(f"\n    Final error: {result.fun:.6f}")
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_result = result
                    print("    ✓ New best result!")
            except Exception as e:
                print(f"\n    ✗ Failed: {str(e)[:50]}")
        
        # Update model
        model.kappa = best_result.x[0]
        model.theta = best_result.x[1]
        model.sigma = best_result.x[2]
        model.rho = best_result.x[3]
        model.v0 = best_result.x[4]
        
        self.one_step_model = model
        self.one_step_result = best_result
        
        elapsed = time.time() - start_time
        print(f"\n\nOne-Step Calibration Complete (Time: {elapsed:.1f}s)")
        print("="*70)
        print(f"  κ (kappa):     {model.kappa:.4f}")
        print(f"  θ (theta):     {model.theta:.4f}")
        print(f"  σ (sigma):     {model.sigma:.4f}")
        print(f"  ρ (rho):       {model.rho:.4f}")
        print(f"  v₀ (v0):       {model.v0:.4f}")
        print(f"  Objective:     {best_result.fun:.6f}")
        
        # Check Feller
        feller = 2 * model.kappa * model.theta
        sigma_sq = model.sigma ** 2
        print(f"\n  Feller: 2κθ = {feller:.4f} vs σ² = {sigma_sq:.4f}")
        print(f"  Satisfied: {'✓' if feller > sigma_sq else '✗'}")
        
        return model
    
    def two_step_calibration(self, option_data, S0):
        """Fast two-step calibration."""
        print("\n" + "="*70)
        print("TWO-STEP CALIBRATION")
        print("="*70)
        
        start_time = time.time()
        
        # Use sample for calibration
        calib_sample = self.select_calibration_sample(option_data, S0, max_options=80)
        
        calibrator = TwoStageHestonCalibrator(r=self.r, q=self.q)
        
        # Stage 1 - with reduced iterations
        print("\nStage 1: Calibrating volatility parameters...")
        stage1_start = time.time()
        
        # Stage 1 bounds - adjusted for high volatility
        bounds = [
            (0.5, 10.0),   # kappa
            (0.01, 1.0),   # theta (up to 100% vol)
            (0.1, 3.0),    # sigma (higher for crypto)
            (0.01, 1.0)    # v0 (up to 100% vol)
        ]
        
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            lambda x: calibrator.stage1_objective(x, calib_sample, S0),
            bounds=bounds,
            maxiter=50,  # Reduced from 100
            popsize=10,  # Reduced from 15
            tol=1e-5,
            seed=42,
            workers=1,
            updating='deferred',
            disp=True
        )
        
        calibrator.stage1_result = {
            'kappa': result.x[0],
            'theta': result.x[1],
            'sigma': result.x[2],
            'v0': result.x[3],
            'objective': result.fun,
            'success': result.success
        }
        
        stage1_time = time.time() - stage1_start
        print(f"\nStage 1 Complete (Time: {stage1_time:.1f}s)")
        print(f"  κ: {result.x[0]:.4f}, θ: {result.x[1]:.4f}, σ: {result.x[2]:.4f}, v₀: {result.x[3]:.4f}")
        
        # Stage 2
        print("\nStage 2: Calibrating correlation...")
        stage2_start = time.time()
        
        stage2_result = calibrator.calibrate_stage2(calib_sample, S0)
        
        stage2_time = time.time() - stage2_start
        
        self.two_step_model = calibrator.model
        self.two_step_result = stage2_result
        
        total_time = time.time() - start_time
        print(f"\n\nTwo-Step Calibration Complete (Total Time: {total_time:.1f}s)")
        print("="*70)
        
        return calibrator.model
    
    def evaluate_pricing_performance(self, option_data, S0):
        """Evaluate both models on full dataset."""
        print("\n" + "="*70)
        print("PRICING PERFORMANCE ON FULL DATASET")
        print("="*70)
        
        results = {
            'one_step': {'errors': [], 'prices': [], 'times': []},
            'two_step': {'errors': [], 'prices': [], 'times': []}
        }
        
        market_prices = []
        
        print(f"\nPricing {len(option_data)} options with both models...")
        
        for i, opt in enumerate(option_data):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(option_data)}", end='\r')
            
            market_price = opt['market_price']
            market_prices.append(market_price)
            
            # One-step
            try:
                t0 = time.time()
                price1 = self.one_step_model.option_price(
                    S0, opt['K'], opt['T'], opt['type']
                )
                t1 = time.time() - t0
                
                if np.isfinite(price1) and price1 > 0:
                    results['one_step']['prices'].append(price1)
                    results['one_step']['times'].append(t1)
                    error1 = abs(price1 - market_price) / market_price * 100
                    results['one_step']['errors'].append(error1)
                else:
                    results['one_step']['prices'].append(np.nan)
                    results['one_step']['errors'].append(np.nan)
                    results['one_step']['times'].append(t1)
            except:
                results['one_step']['prices'].append(np.nan)
                results['one_step']['errors'].append(np.nan)
                results['one_step']['times'].append(0)
            
            # Two-step
            try:
                t0 = time.time()
                price2 = self.two_step_model.option_price(
                    S0, opt['K'], opt['T'], opt['type']
                )
                t1 = time.time() - t0
                
                if np.isfinite(price2) and price2 > 0:
                    results['two_step']['prices'].append(price2)
                    results['two_step']['times'].append(t1)
                    error2 = abs(price2 - market_price) / market_price * 100
                    results['two_step']['errors'].append(error2)
                else:
                    results['two_step']['prices'].append(np.nan)
                    results['two_step']['errors'].append(np.nan)
                    results['two_step']['times'].append(t1)
            except:
                results['two_step']['prices'].append(np.nan)
                results['two_step']['errors'].append(np.nan)
                results['two_step']['times'].append(0)
        
        print(f"\n  Progress: {len(option_data)}/{len(option_data)} - Complete!")
        
        # Calculate statistics
        stats = {}
        for method in ['one_step', 'two_step']:
            errors = np.array(results[method]['errors'])
            valid_errors = errors[~np.isnan(errors)]
            times = np.array(results[method]['times'])
            
            if len(valid_errors) > 0:
                stats[method] = {
                    'mean_error': np.mean(valid_errors),
                    'median_error': np.median(valid_errors),
                    'std_error': np.std(valid_errors),
                    'max_error': np.max(valid_errors),
                    'valid_count': len(valid_errors),
                    'avg_time': np.mean(times) * 1000  # Convert to ms
                }
            else:
                stats[method] = {
                    'mean_error': np.nan,
                    'median_error': np.nan,
                    'std_error': np.nan,
                    'max_error': np.nan,
                    'valid_count': 0,
                    'avg_time': 0
                }
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print("\nOne-Step Method:")
        print(f"  Valid prices:      {stats['one_step']['valid_count']}/{len(option_data)}")
        print(f"  Mean error:        {stats['one_step']['mean_error']:.2f}%")
        print(f"  Median error:      {stats['one_step']['median_error']:.2f}%")
        print(f"  Std error:         {stats['one_step']['std_error']:.2f}%")
        print(f"  Max error:         {stats['one_step']['max_error']:.2f}%")
        print(f"  Avg pricing time:  {stats['one_step']['avg_time']:.2f}ms")
        
        print("\nTwo-Step Method:")
        print(f"  Valid prices:      {stats['two_step']['valid_count']}/{len(option_data)}")
        print(f"  Mean error:        {stats['two_step']['mean_error']:.2f}%")
        print(f"  Median error:      {stats['two_step']['median_error']:.2f}%")
        print(f"  Std error:         {stats['two_step']['std_error']:.2f}%")
        print(f"  Max error:         {stats['two_step']['max_error']:.2f}%")
        print(f"  Avg pricing time:  {stats['two_step']['avg_time']:.2f}ms")
        
        # Winner
        if stats['one_step']['mean_error'] < stats['two_step']['mean_error']:
            diff = stats['two_step']['mean_error'] - stats['one_step']['mean_error']
            print(f"\n{'='*70}")
            print(f"✓ ONE-STEP METHOD WINS by {diff:.2f}% (lower mean error)")
            print(f"{'='*70}")
        else:
            diff = stats['one_step']['mean_error'] - stats['two_step']['mean_error']
            print(f"\n{'='*70}")
            print(f"✓ TWO-STEP METHOD WINS by {diff:.2f}% (lower mean error)")
            print(f"{'='*70}")
        
        return results, stats, market_prices
    
    def plot_comparison(self, option_data, S0, results, stats, market_prices,
                       save_path='plots/calibration_comparison.png'):
        """Create comparison visualization."""
        
        print(f"\nGenerating comparison plots...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        one_prices = np.array(results['one_step']['prices'])
        two_prices = np.array(results['two_step']['prices'])
        market = np.array(market_prices)
        
        one_errors = np.array(results['one_step']['errors'])
        two_errors = np.array(results['two_step']['errors'])
        
        strikes = np.array([opt['K'] for opt in option_data])
        maturities = np.array([opt['T'] for opt in option_data])
        moneyness = strikes / S0
        
        # 1. One-step: Model vs Market
        ax1 = fig.add_subplot(gs[0, 0])
        valid1 = ~np.isnan(one_prices)
        ax1.scatter(market[valid1], one_prices[valid1], alpha=0.4, s=15)
        max_p = max(market[valid1].max(), one_prices[valid1].max())
        ax1.plot([0, max_p], [0, max_p], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Market ($)', fontsize=9)
        ax1.set_ylabel('Model ($)', fontsize=9)
        ax1.set_title('One-Step: Model vs Market', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)
        
        # 2. Two-step: Model vs Market  
        ax2 = fig.add_subplot(gs[0, 1])
        valid2 = ~np.isnan(two_prices)
        ax2.scatter(market[valid2], two_prices[valid2], alpha=0.4, s=15, c='green')
        max_p = max(market[valid2].max(), two_prices[valid2].max())
        ax2.plot([0, max_p], [0, max_p], 'r--', lw=2, label='Perfect')
        ax2.set_xlabel('Market ($)', fontsize=9)
        ax2.set_ylabel('Model ($)', fontsize=9)
        ax2.set_title('Two-Step: Model vs Market', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        
        # 3. Error distributions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(one_errors[~np.isnan(one_errors)], bins=40, alpha=0.5,
                label='One-Step', color='blue', edgecolor='black', range=(0, 100))
        ax3.hist(two_errors[~np.isnan(two_errors)], bins=40, alpha=0.5,
                label='Two-Step', color='green', edgecolor='black', range=(0, 100))
        ax3.axvline(stats['one_step']['mean_error'], color='blue',
                   linestyle='--', lw=2)
        ax3.axvline(stats['two_step']['mean_error'], color='green',
                   linestyle='--', lw=2)
        ax3.set_xlabel('Error (%)', fontsize=9)
        ax3.set_ylabel('Count', fontsize=9)
        ax3.set_title('Error Distribution', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)
        
        # 4-5. Errors by moneyness
        for idx, (method, color, ax_pos) in enumerate([
            ('one_step', 'blue', gs[1, 0]),
            ('two_step', 'green', gs[1, 1])
        ]):
            ax = fig.add_subplot(ax_pos)
            valid = ~np.isnan(results[method]['errors'])
            errors = np.array(results[method]['errors'])
            
            scatter = ax.scatter(moneyness[valid], errors[valid],
                               c=maturities[valid], cmap='plasma',
                               alpha=0.5, s=20, vmin=0, vmax=1)
            ax.axhline(0, color='r', linestyle='--', alpha=0.3)
            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.3)
            ax.set_xlabel('Moneyness', fontsize=9)
            ax.set_ylabel('Error (%)', fontsize=9)
            title = 'One-Step' if method == 'one_step' else 'Two-Step'
            ax.set_title(f'{title}: Error by Moneyness', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-50, 150)
            plt.colorbar(scatter, ax=ax, label='T (years)')
        
        # 6. Direct comparison
        ax6 = fig.add_subplot(gs[1, 2])
        valid_both = valid1 & valid2
        ax6.scatter(one_errors[valid_both], two_errors[valid_both],
                   alpha=0.4, s=20, c=moneyness[valid_both], cmap='coolwarm')
        max_e = min(100, max(one_errors[valid_both].max(), two_errors[valid_both].max()))
        ax6.plot([0, max_e], [0, max_e], 'r--', lw=2)
        ax6.set_xlabel('One-Step Error (%)', fontsize=9)
        ax6.set_ylabel('Two-Step Error (%)', fontsize=9)
        ax6.set_title('Direct Comparison', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.2)
        ax6.set_xlim(0, max_e)
        ax6.set_ylim(0, max_e)
        
        # 7-8. Errors by maturity
        for idx, (method, color, ax_pos) in enumerate([
            ('one_step', 'blue', gs[2, 0]),
            ('two_step', 'green', gs[2, 1])
        ]):
            ax = fig.add_subplot(ax_pos)
            valid = ~np.isnan(results[method]['errors'])
            errors = np.array(results[method]['errors'])
            
            ax.scatter(maturities[valid], errors[valid],
                      alpha=0.5, s=20, color=color)
            ax.axhline(0, color='r', linestyle='--', alpha=0.3)
            ax.set_xlabel('Maturity (years)', fontsize=9)
            ax.set_ylabel('Error (%)', fontsize=9)
            title = 'One-Step' if method == 'one_step' else 'Two-Step'
            ax.set_title(f'{title}: Error by Maturity', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-50, 150)
        
        # 9. Summary table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        table_data = [
            ['Metric', 'One-Step', 'Two-Step'],
            ['Mean Err (%)', f"{stats['one_step']['mean_error']:.1f}",
             f"{stats['two_step']['mean_error']:.1f}"],
            ['Median (%)', f"{stats['one_step']['median_error']:.1f}",
             f"{stats['two_step']['median_error']:.1f}"],
            ['Std (%)', f"{stats['one_step']['std_error']:.1f}",
             f"{stats['two_step']['std_error']:.1f}"],
            ['Max (%)', f"{stats['one_step']['max_error']:.1f}",
             f"{stats['two_step']['max_error']:.1f}"],
            ['Valid', f"{stats['one_step']['valid_count']}",
             f"{stats['two_step']['valid_count']}"],
            ['Time (ms)', f"{stats['one_step']['avg_time']:.1f}",
             f"{stats['two_step']['avg_time']:.1f}"]
        ]
        
        table = ax9.table(cellText=table_data, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#2E7D32')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle('Heston Calibration Comparison: One-Step vs Two-Step',
                    fontsize=13, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
        plt.close()
    
    def plot_volatility_matching(self, option_data, S0,
                                 save_path='plots/volatility_matching.png'):
        """
        Plot volatility matching between Heston expected volatility and market IV.
        
        This uses the option_data directly, computing implied volatility from market prices
        using Black-Scholes inversion if mark_iv is not available.
        """
        
        print("\n" + "="*70)
        print("GENERATING VOLATILITY MATCHING PLOT")
        print("="*70)
        
        # Convert option_data to DataFrame
        df = pd.DataFrame(option_data)
        df['moneyness'] = df['K'] / S0
        
        # Try to get implied volatility data
        # First try to load from original CSV if it has mark_iv
        df_with_iv = None
        try:
            df_csv = pd.read_csv("data/deribit_btc_options_clean.csv")
            if 'mark_iv' in df_csv.columns:
                print("  Using mark_iv from CSV file")
                # Merge with option_data based on strike and maturity
                df_csv['T'] = df_csv.get('T_years', df_csv.get('T', 0))
                df_with_iv = df.merge(
                    df_csv[['strike', 'T', 'mark_iv']], 
                    left_on=['K', 'T'], 
                    right_on=['strike', 'T'],
                    how='left'
                )
        except Exception as e:
            print(f"  Could not load CSV: {e}")
        
        # If no mark_iv available, estimate from market prices using Black-Scholes
        if df_with_iv is None or 'mark_iv' not in df_with_iv.columns or df_with_iv['mark_iv'].isna().all():
            print("  Estimating IV from market prices using Black-Scholes...")
            df_with_iv = df.copy()
            df_with_iv['mark_iv'] = self._estimate_iv_from_prices(df, S0)
        
        # Filter valid data
        df_valid = df_with_iv[
            (df_with_iv['mark_iv'] > 0) & 
            (df_with_iv['mark_iv'].notna()) &
            (df_with_iv['T'] > 0.005) &
            (df_with_iv['moneyness'] > 0.65) &
            (df_with_iv['moneyness'] < 1.5)
        ].copy()
        
        print(f"  Valid options with IV: {len(df_valid)}")
        
        if len(df_valid) < 10:
            print("  ⚠ Not enough data for volatility matching plot")
            # Create a simple placeholder plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 
                   'Insufficient data for volatility matching\n' +
                   'Need market implied volatility data',
                   ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved placeholder to: {save_path}")
            return
        
        # Group by maturity bins
        df_valid['T_bin'] = pd.cut(df_valid['T'], bins=4, duplicates='drop')
        
        # Get 4 maturity groups with most data
        maturity_groups = df_valid.groupby('T_bin').size().nlargest(4).index.tolist()
        maturity_groups = sorted(maturity_groups, key=lambda x: x.mid)
        
        print(f"  Plotting {len(maturity_groups)} maturity groups")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes = axes.flatten()
        
        for idx, T_bin in enumerate(maturity_groups):
            ax = axes[idx]
            
            # Filter data for this maturity
            bin_data = df_valid[df_valid['T_bin'] == T_bin].copy()
            T_mean = bin_data['T'].mean()
            T_days = T_mean * 365
            
            print(f"\n  Group {idx+1}: T={T_mean:.4f} years ({T_days:.1f} days)")
            print(f"    Data points: {len(bin_data)}")
            
            # Market data points
            moneyness_market = bin_data['moneyness'].values
            iv_market = bin_data['mark_iv'].values * 100  # Convert to percentage
            
            # Plot market IVs
            ax.scatter(moneyness_market, iv_market,
                      alpha=0.7, s=80, color='black',
                      label='Market IV', zorder=3,
                      edgecolors='white', linewidths=1)
            
            # Generate model curves
            moneyness_grid = np.linspace(0.65, 1.5, 150)
            one_vols = []
            two_vols = []
            
            for m in moneyness_grid:
                log_m = np.log(m)
                
                # ONE-STEP MODEL
                E_vT_one = self.one_step_model.theta + \
                          (self.one_step_model.v0 - self.one_step_model.theta) * \
                          np.exp(-self.one_step_model.kappa * T_mean)
                
                vol_base_one = np.sqrt(max(E_vT_one, 1e-8))
                
                # Smile adjustments
                adj1 = self.one_step_model.rho * self.one_step_model.sigma * \
                       np.sqrt(T_mean) * log_m / 2.0
                adj2 = (self.one_step_model.sigma**2 * T_mean / 8.0) * \
                       (log_m**2 - 2*log_m)
                
                vol_one = vol_base_one + adj1 + adj2
                one_vols.append(vol_one * 100)
                
                # TWO-STEP MODEL
                E_vT_two = self.two_step_model.theta + \
                          (self.two_step_model.v0 - self.two_step_model.theta) * \
                          np.exp(-self.two_step_model.kappa * T_mean)
                
                vol_base_two = np.sqrt(max(E_vT_two, 1e-8))
                
                adj1_two = self.two_step_model.rho * self.two_step_model.sigma * \
                           np.sqrt(T_mean) * log_m / 2.0
                adj2_two = (self.two_step_model.sigma**2 * T_mean / 8.0) * \
                           (log_m**2 - 2*log_m)
                
                vol_two = vol_base_two + adj1_two + adj2_two
                two_vols.append(vol_two * 100)
            
            # Plot model curves
            ax.plot(moneyness_grid, one_vols, lw=3.5,
                   color='#1f77b4', label='One-Step Heston',
                   alpha=0.85, zorder=2)
            
            ax.plot(moneyness_grid, two_vols, lw=3.5,
                   color='#2ca02c', label='Two-Step Heston',
                   alpha=0.85, linestyle='--', zorder=2)
            
            # Add ATM reference line
            ax.axvline(1.0, color='red', linestyle=':', alpha=0.6, 
                      lw=2.5, label='ATM', zorder=1)
            
            # Formatting
            ax.set_xlabel('Moneyness (K/S₀)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Maturity Group {idx+1}\n' + 
                        f'T = {T_mean:.4f} years ({T_days:.1f} days)',
                        fontsize=12, fontweight='bold', pad=10)
            
            ax.legend(fontsize=10, loc='upper right', framealpha=0.95,
                     edgecolor='gray', fancybox=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            
            ax.set_xlim(0.65, 1.5)
            
            # Dynamic y-axis
            y_min = max(0, min(iv_market) - 10)
            y_max = max(iv_market) + 10
            ax.set_ylim(y_min, y_max)
            
            print(f"    Market IV range: {min(iv_market):.1f}% - {max(iv_market):.1f}%")
        
        fig.suptitle('Heston Model: Expected Volatility vs Market Implied Volatility\n' +
                     'Showing Volatility Smile and Skew Across Strikes',
                    fontsize=15, fontweight='bold', y=0.998)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"\n{'='*70}")
        print(f"✓ Volatility matching plot saved to: {save_path}")
        print(f"{'='*70}\n")
    
    def _estimate_iv_from_prices(self, df, S0):
        """Estimate implied volatility from market prices using Black-Scholes."""
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        ivs = []
        
        for _, row in df.iterrows():
            K = row['K']
            T = row['T']
            price = row['market_price']
            opt_type = row['type']
            
            # Black-Scholes formula
            def bs_price(sigma):
                if sigma <= 0 or T <= 0:
                    return np.inf
                d1 = (np.log(S0/K) + (self.r - self.q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                if opt_type == 'call':
                    return S0*np.exp(-self.q*T)*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
                else:
                    return K*np.exp(-self.r*T)*norm.cdf(-d2) - S0*np.exp(-self.q*T)*norm.cdf(-d1)
            
            # Solve for IV
            try:
                iv = brentq(lambda sig: bs_price(sig) - price, 0.01, 3.0)
                ivs.append(iv)
            except:
                ivs.append(np.nan)
        
        return ivs


def main():
    """Main execution."""
    
    print("="*70)
    print("FAST HESTON CALIBRATION COMPARISON")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Load data
    print("\nLoading data...")
    option_data, S0 = parse_option_data("data/deribit_btc_options_clean.csv")
    print(f"✓ Loaded {len(option_data)} options (S₀ = ${S0:.2f})")
    
    # Initialize
    comparison = FastCalibrationComparison(r=0.01, q=0.0)
    
    # Calibrate
    comparison.one_step_calibration(option_data, S0)
    comparison.two_step_calibration(option_data, S0)
    
    # Evaluate
    results, stats, market_prices = comparison.evaluate_pricing_performance(
        option_data, S0
    )
    
    # Plot
    comparison.plot_comparison(option_data, S0, results, stats, market_prices)
    comparison.plot_volatility_matching(option_data, S0)
    
    # Save results
    print("\nSaving detailed results...")
    results_df = pd.DataFrame({
        'strike': [opt['K'] for opt in option_data],
        'maturity': [opt['T'] for opt in option_data],
        'type': [opt['type'] for opt in option_data],
        'market_price': market_prices,
        'one_step_price': results['one_step']['prices'],
        'two_step_price': results['two_step']['prices'],
        'one_step_error_pct': results['one_step']['errors'],
        'two_step_error_pct': results['two_step']['errors']
    })
    
    results_df.to_csv('results/calibration_comparison.csv', index=False)
    print("✓ Saved to: results/calibration_comparison.csv")
    
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"COMPLETE - Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
two_stage_heston_calibration.py - Two-Stage Heston Model Calibration

Stage 1: Calibrate volatility parameters (kappa, theta, sigma, v0) 
         to match implied volatility surface
Stage 2: Calibrate correlation (rho) using option prices
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from heston import HestonModel, parse_option_data


class TwoStageHestonCalibrator:
    """
    Two-stage Heston model calibration:
    1. First stage: Match IV surface (volatility parameters)
    2. Second stage: Match option prices (correlation parameter)
    """
    
    def __init__(self, r: float = 0.01, q: float = 0.0):
        """
        Initialize calibrator.
        
        Parameters:
        -----------
        r : float
            Risk-free rate
        q : float
            Dividend/convenience yield
        """
        self.r = r
        self.q = q
        self.model = None
        self.stage1_result = None
        self.stage2_result = None
        
    def black_scholes_price(self, S: float, K: float, T: float,
                           sigma: float, option_type: str = "call") -> float:
        """Black-Scholes pricing formula."""
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return intrinsic
        
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(-self.q * T) * norm.cdf(-d1)
        
        return float(price)
    
    def heston_instantaneous_vol(self, T: float, kappa: float, 
                                theta: float, v0: float) -> float:
        """
        Calculate expected instantaneous volatility at time T under Heston.
        E[v_T] = theta + (v0 - theta) * exp(-kappa * T)
        """
        return theta + (v0 - theta) * np.exp(-kappa * T)
    
    def heston_integrated_variance(self, T: float, kappa: float,
                                   theta: float, v0: float) -> float:
        """
        Calculate integrated variance over [0, T] under Heston.
        This gives us the "average" variance for BS approximation.
        
        Var = (1/T) * E[∫₀ᵀ v_s ds]
        """
        if T <= 0:
            return v0
        
        if abs(kappa) < 1e-8:
            # If kappa ≈ 0, variance doesn't mean-revert
            return v0
        
        # Integrated variance formula
        integrated_var = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
        avg_var = integrated_var / T
        
        return avg_var
    
    def stage1_objective(self, params, option_data, S0):
        """
        Stage 1: Match implied volatility surface.
        
        Minimize squared error between:
        - Market IV
        - Heston model's expected integrated variance (converted to vol)
        
        Parameters to calibrate: [kappa, theta, sigma, v0]
        Fixed: rho = -0.5 (typical for equities/crypto)
        """
        kappa, theta, sigma, v0 = params
        
        # Feller condition penalty
        feller_violation = max(0.0, sigma**2 - 2.0 * kappa * theta)
        penalty = 1e6 * feller_violation
        
        # Parameter sanity penalty
        if v0 < 0 or theta < 0 or kappa < 0 or sigma < 0:
            return 1e10
        
        errors = []
        for opt in option_data:
            # Get market IV (already in the data as mark_iv)
            market_iv = opt.get('mark_iv', 0)
            if market_iv <= 0:
                continue
            
            # Calculate Heston's expected integrated variance
            heston_var = self.heston_integrated_variance(
                opt['T'], kappa, theta, v0
            )
            heston_vol = np.sqrt(heston_var)
            
            # Add vol-of-vol adjustment (approximation)
            # Higher sigma means more convexity in the smile
            K_over_S = opt['K'] / S0
            moneyness_adj = sigma * np.sqrt(opt['T']) * (K_over_S - 1.0) / 4.0
            heston_vol_adjusted = heston_vol + moneyness_adj
            
            # Squared error in volatility space
            error = (heston_vol_adjusted - market_iv) ** 2
            errors.append(error)
        
        mse = np.mean(errors) if errors else 1e10
        return mse + penalty
    
    def stage2_objective(self, rho, kappa, theta, sigma, v0,
                        option_data, S0):
        """
        Stage 2: Match option prices with fixed volatility parameters.
        
        Only calibrate: rho (correlation)
        Fixed: kappa, theta, sigma, v0 (from stage 1)
        """
        # Create Heston model with stage 1 parameters + current rho
        model = HestonModel(
            kappa=kappa, theta=theta, sigma=sigma,
            rho=rho, v0=v0, r=self.r, q=self.q
        )
        
        errors = []
        weights = []
        
        for opt in option_data:
            try:
                # Model price
                model_price = model.option_price(
                    S0, opt['K'], opt['T'], opt['type']
                )
                
                # Market price
                market_price = opt['market_price']
                
                # Relative error (better for options with different price scales)
                rel_error = ((model_price - market_price) / market_price) ** 2
                
                # Weight by open interest and inverse of spread
                weight = 1.0
                if 'open_interest' in opt and opt['open_interest'] > 0:
                    weight *= np.log1p(opt['open_interest'])
                if 'spread_pct' in opt and opt['spread_pct'] > 0:
                    weight *= 1.0 / (1.0 + opt['spread_pct'])
                
                errors.append(rel_error)
                weights.append(weight)
                
            except Exception as e:
                continue
        
        if not errors:
            return 1e10
        
        # Weighted mean squared error
        weighted_mse = np.average(errors, weights=weights)
        return weighted_mse
    
    def calibrate_stage1(self, option_data: list, S0: float,
                        method: str = 'differential_evolution'):
        """
        Stage 1: Calibrate volatility parameters to IV surface.
        
        Parameters:
        -----------
        option_data : list
            List of option data dictionaries
        S0 : float
            Spot price
        method : str
            'differential_evolution' or 'minimize'
        
        Returns:
        --------
        dict
            Calibrated parameters
        """
        print("\n" + "="*60)
        print("STAGE 1: Calibrating Volatility Parameters to IV Surface")
        print("="*60)
        
        # Bounds: [kappa, theta, sigma, v0]
        bounds = [
            (0.5, 5.0),      # kappa: mean reversion speed
            (0.01, 0.5),     # theta: long-term variance
            (0.1, 2.0),      # sigma: vol of vol
            (0.01, 0.5)      # v0: initial variance
        ]
        
        if method == 'differential_evolution':
            print("Using Differential Evolution (global optimization)...")
            result = differential_evolution(
                lambda x: self.stage1_objective(x, option_data, S0),
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                workers=1,
                updating='deferred'
            )
        else:
            print("Using L-BFGS-B (local optimization)...")
            initial = [2.0, 0.04, 0.6, 0.04]
            result = minimize(
                lambda x: self.stage1_objective(x, option_data, S0),
                x0=initial,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 300}
            )
        
        self.stage1_result = {
            'kappa': result.x[0],
            'theta': result.x[1],
            'sigma': result.x[2],
            'v0': result.x[3],
            'objective': result.fun,
            'success': result.success
        }
        
        print("\nStage 1 Results:")
        print(f"  κ (kappa): {self.stage1_result['kappa']:.4f}")
        print(f"  θ (theta): {self.stage1_result['theta']:.4f}")
        print(f"  σ (sigma): {self.stage1_result['sigma']:.4f}")
        print(f"  v₀ (v0):   {self.stage1_result['v0']:.4f}")
        print(f"  MSE (IV):  {self.stage1_result['objective']:.6f}")
        print(f"  Success:   {self.stage1_result['success']}")
        
        # Check Feller condition
        feller = 2 * self.stage1_result['kappa'] * self.stage1_result['theta']
        sigma_sq = self.stage1_result['sigma'] ** 2
        print(f"\n  Feller condition: 2κθ = {feller:.4f} vs σ² = {sigma_sq:.4f}")
        print(f"  Satisfied: {feller > sigma_sq}")
        
        return self.stage1_result
    
    def calibrate_stage2(self, option_data: list, S0: float):
        """
        Stage 2: Calibrate correlation parameter to option prices.
        
        Parameters:
        -----------
        option_data : list
            List of option data dictionaries
        S0 : float
            Spot price
        
        Returns:
        --------
        dict
            Full calibrated parameters
        """
        if self.stage1_result is None:
            raise ValueError("Must run stage 1 calibration first!")
        
        print("\n" + "="*60)
        print("STAGE 2: Calibrating Correlation to Option Prices")
        print("="*60)
        
        # Extract stage 1 parameters
        kappa = self.stage1_result['kappa']
        theta = self.stage1_result['theta']
        sigma = self.stage1_result['sigma']
        v0 = self.stage1_result['v0']
        
        # Grid search for rho (coarse)
        print("\nCoarse grid search for rho...")
        rho_grid = np.linspace(-0.9, 0.5, 15)
        best_rho = -0.5
        best_error = float('inf')
        
        for rho in rho_grid:
            error = self.stage2_objective(
                rho, kappa, theta, sigma, v0, option_data, S0
            )
            if error < best_error:
                best_error = error
                best_rho = rho
        
        print(f"Coarse grid best: rho = {best_rho:.3f}, error = {best_error:.6f}")
        
        # Fine optimization around best value
        print("\nFine optimization around best rho...")
        result = minimize(
            lambda x: self.stage2_objective(
                x[0], kappa, theta, sigma, v0, option_data, S0
            ),
            x0=[best_rho],
            bounds=[(-0.99, 0.99)],
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        self.stage2_result = {
            'rho': result.x[0],
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'v0': v0,
            'objective': result.fun,
            'success': result.success
        }
        
        print("\nStage 2 Results:")
        print(f"  ρ (rho):     {self.stage2_result['rho']:.4f}")
        print(f"  MSE (Price): {self.stage2_result['objective']:.6f}")
        print(f"  Success:     {self.stage2_result['success']}")
        
        # Create final calibrated model
        self.model = HestonModel(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=self.stage2_result['rho'],
            v0=v0,
            r=self.r,
            q=self.q
        )
        
        return self.stage2_result
    
    def evaluate_fit(self, option_data: list, S0: float,
                    save_path: str = None):
        """
        Evaluate and visualize the calibration fit.
        
        Parameters:
        -----------
        option_data : list
            List of option data dictionaries
        S0 : float
            Spot price
        save_path : str
            Path to save figure (optional)
        """
        if self.model is None:
            raise ValueError("Must complete calibration first!")
        
        print("\n" + "="*60)
        print("EVALUATING FIT QUALITY")
        print("="*60)
        
        # Calculate model prices
        model_prices = []
        market_prices = []
        strikes = []
        maturities = []
        types = []
        
        for opt in option_data:
            try:
                model_price = self.model.option_price(
                    S0, opt['K'], opt['T'], opt['type']
                )
                model_prices.append(model_price)
                market_prices.append(opt['market_price'])
                strikes.append(opt['K'])
                maturities.append(opt['T'])
                types.append(opt['type'])
            except:
                continue
        
        model_prices = np.array(model_prices)
        market_prices = np.array(market_prices)
        
        # Calculate errors
        abs_errors = np.abs(model_prices - market_prices)
        rel_errors = abs_errors / market_prices * 100
        
        print(f"\nPrice Fit Statistics:")
        print(f"  Mean Absolute Error: ${np.mean(abs_errors):.2f}")
        print(f"  Median Absolute Error: ${np.median(abs_errors):.2f}")
        print(f"  Mean Relative Error: {np.mean(rel_errors):.2f}%")
        print(f"  Median Relative Error: {np.median(rel_errors):.2f}%")
        print(f"  Max Absolute Error: ${np.max(abs_errors):.2f}")
        print(f"  RMSE: ${np.sqrt(np.mean(abs_errors**2)):.2f}")
        
        # Separate by option type
        calls = [i for i, t in enumerate(types) if t == 'call']
        puts = [i for i, t in enumerate(types) if t == 'put']
        
        print(f"\nBy Option Type:")
        print(f"  Calls - Mean Rel Error: {np.mean(rel_errors[calls]):.2f}%")
        print(f"  Puts  - Mean Rel Error: {np.mean(rel_errors[puts]):.2f}%")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Model vs Market prices
        ax = axes[0, 0]
        ax.scatter(market_prices, model_prices, alpha=0.5, s=20)
        max_price = max(market_prices.max(), model_prices.max())
        ax.plot([0, max_price], [0, max_price], 'r--', label='Perfect fit')
        ax.set_xlabel('Market Price (USD)', fontsize=11)
        ax.set_ylabel('Model Price (USD)', fontsize=11)
        ax.set_title('Model vs Market Prices', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Relative errors by moneyness
        ax = axes[0, 1]
        moneyness = np.array(strikes) / S0
        scatter = ax.scatter(moneyness, rel_errors, c=maturities,
                            cmap='viridis', alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Moneyness (K/S₀)', fontsize=11)
        ax.set_ylabel('Relative Error (%)', fontsize=11)
        ax.set_title('Pricing Errors by Moneyness', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Maturity (years)')
        
        # 3. Error distribution
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
        
        # 4. Errors by maturity
        ax = axes[1, 1]
        ax.scatter(maturities, rel_errors, alpha=0.5, s=30)
        ax.set_xlabel('Time to Maturity (years)', fontsize=11)
        ax.set_ylabel('Relative Error (%)', fontsize=11)
        ax.set_title('Pricing Errors by Maturity', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        plt.show()
        
        return {
            'mae': float(np.mean(abs_errors)),
            'mape': float(np.mean(rel_errors)),
            'rmse': float(np.sqrt(np.mean(abs_errors**2)))
        }


def main():
    """Main function demonstrating two-stage calibration."""
    
    # Load data
    print("Loading option data...")
    option_data, S0 = parse_option_data("../data/deribit_btc_options_clean.csv")
    
    print(f"Loaded {len(option_data)} options")
    print(f"Spot price: ${S0:.2f}")
    
    # Create calibrator
    calibrator = TwoStageHestonCalibrator(r=0.01, q=0.0)
    
    # Stage 1: Calibrate volatility parameters to IV surface
    stage1_result = calibrator.calibrate_stage1(
        option_data, S0, method='differential_evolution'
    )
    
    # Stage 2: Calibrate correlation to option prices
    stage2_result = calibrator.calibrate_stage2(option_data, S0)
    
    # Evaluate fit
    fit_stats = calibrator.evaluate_fit(
        option_data, S0, save_path='plots/calibration_fit.png'
    )
    
    # Print final calibrated parameters
    print("\n" + "="*60)
    print("FINAL CALIBRATED PARAMETERS")
    print("="*60)
    print(f"κ (mean reversion):  {stage2_result['kappa']:.4f}")
    print(f"θ (long-term var):   {stage2_result['theta']:.4f}")
    print(f"σ (vol of vol):      {stage2_result['sigma']:.4f}")
    print(f"ρ (correlation):     {stage2_result['rho']:.4f}")
    print(f"v₀ (initial var):    {stage2_result['v0']:.4f}")
    print(f"\nLong-term volatility: {np.sqrt(stage2_result['theta'])*100:.2f}%")
    print(f"Current volatility:   {np.sqrt(stage2_result['v0'])*100:.2f}%")
    
    # Test pricing
    print("\n" + "="*60)
    print("SAMPLE PRICING RESULTS")
    print("="*60)
    
    for i in [0, 10, 50, 100]:
        if i >= len(option_data):
            break
        opt = option_data[i]
        model_price = calibrator.model.option_price(
            S0, opt['K'], opt['T'], opt['type']
        )
        market_price = opt['market_price']
        error = (model_price - market_price) / market_price * 100
        
        print(f"\nOption {i+1}: {opt['type'].upper()}")
        print(f"  Strike: ${opt['K']:.0f}, Maturity: {opt['T']:.4f} years")
        print(f"  Market: ${market_price:.2f}")
        print(f"  Model:  ${model_price:.2f}")
        print(f"  Error:  {error:+.2f}%")


if __name__ == "__main__":
    main()
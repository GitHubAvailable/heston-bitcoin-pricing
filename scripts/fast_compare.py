# -*- coding: utf-8 -*-
"""
快速版本的校准对比 - 优化性能和数值稳定性
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

from heston import parse_option_data


class FastHestonModel:
    """优化的Heston模型 - 更好的数值稳定性"""
    
    def __init__(self, kappa, theta, sigma, rho, v0, r=0.01, q=0.0):
        self.kappa = max(kappa, 0.01)
        self.theta = max(theta, 1e-6)
        self.sigma = max(sigma, 0.01)
        self.rho = np.clip(rho, -0.99, 0.99)
        self.v0 = max(v0, 1e-6)
        self.r = r
        self.q = q
    
    def update_params(self, params):
        """更新参数"""
        self.kappa = max(params[0], 0.01)
        self.theta = max(params[1], 1e-6)
        self.sigma = max(params[2], 0.01)
        self.rho = np.clip(params[3], -0.99, 0.99)
        self.v0 = max(params[4], 1e-6)
    
    def black_scholes_price(self, S, K, T, sigma, option_type='call'):
        """BS定价 - 用于快速近似"""
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (self.r - self.q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            return S*np.exp(-self.q*T)*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-self.r*T)*norm.cdf(-d2) - S*np.exp(-self.q*T)*norm.cdf(-d1)
    
    def heston_price_approximation(self, S0, K, T, option_type='call'):
        """
        Heston近似定价 - 使用积分方差的BS公式
        比完整的CF积分快得多
        """
        # 计算期望积分方差
        if T <= 0:
            return self.black_scholes_price(S0, K, T, np.sqrt(self.v0), option_type)
        
        # E[∫v_s ds] / T
        if abs(self.kappa) < 1e-8:
            avg_var = self.v0
        else:
            integrated = self.theta * T + (self.v0 - self.theta) * (1 - np.exp(-self.kappa*T)) / self.kappa
            avg_var = integrated / T
        
        # 添加偏度调整
        moneyness = K / S0
        skew_adj = self.rho * self.sigma * np.sqrt(T) * (moneyness - 1) / 2
        
        approx_vol = np.sqrt(max(avg_var, 1e-6)) + skew_adj
        approx_vol = max(approx_vol, 0.01)  # 防止负波动率
        
        return self.black_scholes_price(S0, K, T, approx_vol, option_type)


def select_calibration_options(option_data, S0, max_per_expiry=15):
    """智能选择校准用期权 - 使用更多期权以提高精度"""
    df = pd.DataFrame(option_data)
    df['moneyness'] = df['K'] / S0
    df['abs_log_moneyness'] = np.abs(np.log(df['moneyness']))
    
    selected = []
    
    # 按到期日分组
    for T in sorted(df['T'].unique()):
        df_T = df[df['T'] == T].copy()
        
        # 优先选择ATM附近的期权,但也保留一些OTM期权
        df_T['priority'] = df_T['abs_log_moneyness']
        
        # 每个到期日选择Call和Put
        n_per_type = max_per_expiry // 2
        
        calls = df_T[df_T['type'] == 'call'].nsmallest(n_per_type, 'priority')
        puts = df_T[df_T['type'] == 'put'].nsmallest(n_per_type, 'priority')
        
        selected.extend(calls.to_dict('records'))
        selected.extend(puts.to_dict('records'))
    
    return selected


def calibrate_price_based(option_data, S0, r=0.01, q=0.0):
    """Method 1: Price-based calibration"""
    print("\n" + "="*70)
    print("Method 1: Direct Price Calibration")
    print("="*70)
    
    model = FastHestonModel(2.0, 0.04, 0.5, -0.3, 0.04, r, q)
    
    def objective(params):
        model.update_params(params)
        
        # Feller condition penalty
        feller_penalty = max(0, model.sigma**2 - 2*model.kappa*model.theta) * 1e5
        
        # Parameter boundary penalty
        if model.v0 > 1.0 or model.theta > 1.0:
            return 1e10
        
        errors = []
        for opt in option_data:
            try:
                model_price = model.heston_price_approximation(
                    S0, opt['K'], opt['T'], opt['type']
                )
                market_price = opt['market_price']
                
                # Relative error
                rel_error = ((model_price - market_price) / market_price) ** 2
                
                # Weight (reduce impact of extreme options)
                weight = 1.0 / (1.0 + opt.get('spread_pct', 0))
                
                errors.append(weight * rel_error)
            except:
                continue
        
        if not errors:
            return 1e10
        
        return np.mean(errors) + feller_penalty
    
    # Use differential evolution for global optimization
    bounds = [
        (0.5, 5.0),    # kappa
        (0.01, 0.3),   # theta
        (0.1, 1.5),    # sigma
        (-0.9, 0.5),   # rho
        (0.01, 0.3)    # v0
    ]
    
    print(f"Starting calibration with {len(option_data)} options...")
    
    result = differential_evolution(
        objective, bounds,
        maxiter=80,  # Increase iterations for better accuracy
        popsize=12,
        tol=1e-5,
        seed=42,
        workers=1
    )
    
    model.update_params(result.x)
    
    print(f"\nCalibration complete!")
    print(f"  kappa: {model.kappa:.4f}")
    print(f"  theta: {model.theta:.4f}")
    print(f"  sigma: {model.sigma:.4f}")
    print(f"  rho:   {model.rho:.4f}")
    print(f"  v0:    {model.v0:.4f}")
    print(f"  Objective: {result.fun:.6f}")
    
    return model, result


def calibrate_two_stage(option_data, S0, r=0.01, q=0.0):
    """Method 2: Two-stage calibration"""
    print("\n" + "="*70)
    print("Method 2: Two-Stage Calibration")
    print("="*70)
    
    # Stage 1: IV calibration
    print("\nStage 1: Calibrating volatility parameters...")
    
    def stage1_objective(params):
        kappa, theta, sigma, v0 = params
        
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0:
            return 1e10
        
        feller_penalty = max(0, sigma**2 - 2*kappa*theta) * 1e5
        
        errors = []
        for opt in option_data:
            market_iv = opt.get('mark_iv', 0)
            if market_iv <= 0:
                continue
            
            T = opt['T']
            
            # Calculate model IV (integrated variance)
            if abs(kappa) < 1e-8:
                model_var = v0
            else:
                integrated = theta * T + (v0 - theta) * (1 - np.exp(-kappa*T)) / kappa
                model_var = integrated / T
            
            model_vol = np.sqrt(max(model_var, 1e-6))
            
            # Moneyness adjustment
            moneyness_adj = sigma * np.sqrt(T) * (opt['K']/S0 - 1) / 4
            model_vol_adj = model_vol + moneyness_adj
            
            error = (model_vol_adj - market_iv) ** 2
            errors.append(error)
        
        return np.mean(errors) + feller_penalty if errors else 1e10
    
    bounds1 = [(0.5, 5), (0.01, 0.3), (0.1, 1.5), (0.01, 0.3)]
    
    result1 = differential_evolution(
        stage1_objective, bounds1,
        maxiter=50, popsize=12, tol=1e-5, seed=42
    )
    
    kappa, theta, sigma, v0 = result1.x
    
    print(f"Stage 1 complete:")
    print(f"  kappa: {kappa:.4f}, theta: {theta:.4f}, sigma: {sigma:.4f}, v0: {v0:.4f}")
    
    # Stage 2: Calibrate rho
    print("\nStage 2: Calibrating correlation...")
    
    def stage2_objective(rho):
        model = FastHestonModel(kappa, theta, sigma, rho, v0, r, q)
        
        errors = []
        for opt in option_data:
            try:
                model_price = model.heston_price_approximation(
                    S0, opt['K'], opt['T'], opt['type']
                )
                market_price = opt['market_price']
                rel_error = ((model_price - market_price) / market_price) ** 2
                errors.append(rel_error)
            except:
                continue
        
        return np.mean(errors) if errors else 1e10
    
    result2 = minimize(
        lambda x: stage2_objective(x[0]),
        x0=[-0.3],
        bounds=[(-0.9, 0.5)],
        method='L-BFGS-B'
    )
    
    rho = result2.x[0]
    
    print(f"Stage 2 complete:")
    print(f"  rho: {rho:.4f}")
    
    model = FastHestonModel(kappa, theta, sigma, rho, v0, r, q)
    
    return model, (result1, result2)


def evaluate_models(model1, model2, option_data, S0):
    """Evaluate both models"""
    print("\n" + "="*70)
    print("Evaluating model pricing accuracy...")
    print("="*70)
    
    results = []
    
    for opt in option_data:
        try:
            # Method 1
            price1 = model1.heston_price_approximation(
                S0, opt['K'], opt['T'], opt['type']
            )
            error1 = price1 - opt['market_price']
            rel_error1 = (error1 / opt['market_price']) * 100
            
            # Method 2
            price2 = model2.heston_price_approximation(
                S0, opt['K'], opt['T'], opt['type']
            )
            error2 = price2 - opt['market_price']
            rel_error2 = (error2 / opt['market_price']) * 100
            
            results.append({
                'K': opt['K'],
                'T': opt['T'],
                'type': opt['type'],
                'moneyness': opt['K'] / S0,
                'market_price': opt['market_price'],
                'market_iv': opt.get('mark_iv', np.nan),
                'price1': price1,
                'error1': error1,
                'rel_error1': rel_error1,
                'price2': price2,
                'error2': error2,
                'rel_error2': rel_error2,
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    print(f"\nSuccessfully evaluated {len(df)} options")
    
    print("\nMethod 1 (Price-based) Statistics:")
    print(f"  MAE:  ${df['error1'].abs().mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df['error1']**2).mean()):.2f}")
    print(f"  MAPE: {df['rel_error1'].abs().mean():.2f}%")
    
    print("\nMethod 2 (Two-stage) Statistics:")
    print(f"  MAE:  ${df['error2'].abs().mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df['error2']**2).mean()):.2f}")
    print(f"  MAPE: {df['rel_error2'].abs().mean():.2f}%")
    
    return df


def plot_comparison(df, model1, model2, S0, save_path='plots/fast_comparison.png'):
    """Generate comparison plots"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Pricing comparison
    ax1 = fig.add_subplot(gs[0, 0])
    max_price = df['market_price'].max()
    ax1.scatter(df['market_price'], df['price1'], alpha=0.4, s=15, label='Method 1', c='blue')
    ax1.scatter(df['market_price'], df['price2'], alpha=0.4, s=15, label='Method 2', c='red')
    ax1.plot([0, max_price], [0, max_price], 'k--', lw=1, label='Perfect fit')
    ax1.set_xlabel('Market Price (USD)', fontsize=10)
    ax1.set_ylabel('Model Price (USD)', fontsize=10)
    ax1.set_title('Model Pricing vs Market Price', fontweight='bold', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Error distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['rel_error1'], bins=40, alpha=0.6, label='Method 1', color='blue', edgecolor='black')
    ax2.hist(df['rel_error2'], bins=40, alpha=0.6, label='Method 2', color='red', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', lw=1)
    ax2.set_xlabel('Relative Error (%)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Pricing Error Distribution', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. IV comparison - Call
    ax3 = fig.add_subplot(gs[0, 2])
    calls = df[df['type'] == 'call']
    ax3.scatter(calls['moneyness'], calls['market_iv']*100, 
               alpha=0.5, s=25, c='green', label='Market IV')
    ax3.axhline(model1.v0**0.5*100, color='blue', lw=2, 
               label=f'Method 1: sqrt(v0)={model1.v0**0.5*100:.1f}%')
    ax3.axhline(model2.v0**0.5*100, color='red', lw=2,
               label=f'Method 2: sqrt(v0)={model2.v0**0.5*100:.1f}%')
    ax3.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Moneyness (K/S0)', fontsize=10)
    ax3.set_ylabel('Volatility (%)', fontsize=10)
    ax3.set_title('Implied Volatility - Call Options', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Error vs Moneyness
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['moneyness'], df['rel_error1'].abs(), alpha=0.4, s=15, label='Method 1', c='blue')
    ax4.scatter(df['moneyness'], df['rel_error2'].abs(), alpha=0.4, s=15, label='Method 2', c='red')
    ax4.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Moneyness (K/S0)', fontsize=10)
    ax4.set_ylabel('|Relative Error| (%)', fontsize=10)
    ax4.set_title('Pricing Error vs Moneyness', fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. IV comparison - Put
    ax5 = fig.add_subplot(gs[1, 1])
    puts = df[df['type'] == 'put']
    ax5.scatter(puts['moneyness'], puts['market_iv']*100,
               alpha=0.5, s=25, c='green', marker='^', label='Market IV')
    ax5.axhline(model1.v0**0.5*100, color='blue', lw=2, 
               label=f'Method 1: sqrt(v0)={model1.v0**0.5*100:.1f}%')
    ax5.axhline(model2.v0**0.5*100, color='red', lw=2,
               label=f'Method 2: sqrt(v0)={model2.v0**0.5*100:.1f}%')
    ax5.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Moneyness (K/S0)', fontsize=10)
    ax5.set_ylabel('Volatility (%)', fontsize=10)
    ax5.set_title('Implied Volatility - Put Options', fontweight='bold', fontsize=11)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    
    # 6. Parameter comparison table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    param_text = [
        ['Parameter', 'Method 1', 'Method 2'],
        ['kappa', f'{model1.kappa:.4f}', f'{model2.kappa:.4f}'],
        ['theta', f'{model1.theta:.4f}', f'{model2.theta:.4f}'],
        ['sigma', f'{model1.sigma:.4f}', f'{model2.sigma:.4f}'],
        ['rho', f'{model1.rho:.4f}', f'{model2.rho:.4f}'],
        ['v0', f'{model1.v0:.4f}', f'{model2.v0:.4f}'],
        ['', '', ''],
        ['sqrt(v0) %', f'{model1.v0**0.5*100:.2f}', f'{model2.v0**0.5*100:.2f}'],
        ['sqrt(theta) %', f'{model1.theta**0.5*100:.2f}', f'{model2.theta**0.5*100:.2f}'],
    ]
    
    table = ax6.table(cellText=param_text, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Calibrated Parameters', fontweight='bold', pad=20, fontsize=12)
    
    plt.suptitle('Heston Model Calibration Methods Comparison', fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.show()


def main():
    import os
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("Loading option data...")
    all_options, S0 = parse_option_data("data/deribit_btc_options_clean.csv")
    print(f"Loaded {len(all_options)} options, S0 = ${S0:.2f}")
    
    # Select calibration options - use more options for better accuracy
    calib_options = select_calibration_options(all_options, S0, max_per_expiry=15)
    print(f"Selected {len(calib_options)} options for calibration")
    print(f"This represents {len(calib_options)/len(all_options)*100:.1f}% of all options")
    
    # Calibration
    model1, result1 = calibrate_price_based(calib_options, S0)
    model2, result2 = calibrate_two_stage(calib_options, S0)
    
    # Evaluate on ALL options
    print(f"\nEvaluating on ALL {len(all_options)} options...")
    df_results = evaluate_models(model1, model2, all_options, S0)
    
    # Save results
    df_results.to_csv('results/fast_comparison.csv', index=False)
    print("Results saved: results/fast_comparison.csv")
    
    # Generate plots
    plot_comparison(df_results, model1, model2, S0)
    
    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\nCalibration dataset: {len(calib_options)} options")
    print(f"Evaluation dataset: {len(all_options)} options")
    
    print("\nMethod 1 (Price-based calibration):")
    print(f"  Parameters: kappa={model1.kappa:.4f}, theta={model1.theta:.4f}, ")
    print(f"              sigma={model1.sigma:.4f}, rho={model1.rho:.4f}, v0={model1.v0:.4f}")
    print(f"  Current vol: {model1.v0**0.5*100:.2f}%")
    print(f"  Long-term vol: {model1.theta**0.5*100:.2f}%")
    
    print("\nMethod 2 (Two-stage calibration):")
    print(f"  Parameters: kappa={model2.kappa:.4f}, theta={model2.theta:.4f}, ")
    print(f"              sigma={model2.sigma:.4f}, rho={model2.rho:.4f}, v0={model2.v0:.4f}")
    print(f"  Current vol: {model2.v0**0.5*100:.2f}%")
    print(f"  Long-term vol: {model2.theta**0.5*100:.2f}%")
    
    # Compare with market IV
    market_iv_mean = df_results['market_iv'].mean() * 100
    print(f"\nMarket average IV: {market_iv_mean:.2f}%")
    print(f"Method 1 deviation: {abs(model1.v0**0.5*100 - market_iv_mean):.2f}%")
    print(f"Method 2 deviation: {abs(model2.v0**0.5*100 - market_iv_mean):.2f}%")
    
    # Better method
    mae1 = df_results['error1'].abs().mean()
    mae2 = df_results['error2'].abs().mean()
    
    print("\n" + "="*70)
    if mae1 < mae2:
        print("CONCLUSION: Method 1 (Price-based) achieves lower pricing errors")
    else:
        print("CONCLUSION: Method 2 (Two-stage) achieves lower pricing errors")
    print("="*70)
    
    print("\nComplete!")


if __name__ == "__main__":
    main()
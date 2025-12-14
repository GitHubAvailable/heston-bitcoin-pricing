# -*- coding: utf-8 -*-
"""
compare_calib_fixed.py - 修复后的Heston校准对比脚本

修复问题：
1. mark_iv已经是百分比形式（如52.08代表52.08%），不需要再乘以100
2. 模型波动率需要正确计算
3. 显示所有maturity的数据
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')


class HestonParams:
    """Heston模型参数容器"""
    def __init__(self, kappa=2.0, theta=0.25, sigma=1.0, rho=-0.7, v0=0.30):
        self.kappa = kappa  # 均值回归速度
        self.theta = theta  # 长期方差 (波动率² = theta, 所以波动率 = sqrt(theta))
        self.sigma = sigma  # 波动率的波动率
        self.rho = rho      # 相关系数
        self.v0 = v0        # 初始方差


def parse_option_data(path):
    """解析期权数据"""
    df = pd.read_csv(path)
    option_data = []
    for _, row in df.iterrows():
        option_data.append({
            "K": float(row["strike"]),
            "T": float(row["T_years"]),
            "market_price": float(row["market_price_usd"]),
            "type": row["type"],
            "mark_iv": float(row["mark_iv"]) / 100.0,  # 转换为小数形式
            "spread_pct": float(row["spread_pct"]) if "spread_pct" in row and pd.notna(row["spread_pct"]) else None,
            "open_interest": float(row["open_interest"]) if "open_interest" in row and pd.notna(row["open_interest"]) else 0,
            "expiry_date": row["expiry_date"]
        })
    return option_data, float(df["S0"].iloc[0])


def calculate_heston_iv(params, T, moneyness_array):
    """
    计算Heston模型的隐含波动率近似
    
    使用公式：
        E[v_T] = θ + (v₀ - θ) * exp(-κT)
        σ_IV ≈ √E[v_T] + 偏度调整 + 微笑调整
    
    返回百分比形式（如50代表50%）
    """
    vols = []
    
    for m in moneyness_array:
        log_m = np.log(m)
        
        # 计算到期时T的期望方差
        E_vT = params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * T)
        
        # 基础波动率
        vol_base = np.sqrt(max(E_vT, 1e-8))
        
        # 一阶调整（偏度）- 由相关系数引起
        adj1 = params.rho * params.sigma * np.sqrt(T) * log_m / 2.0
        
        # 二阶调整（微笑）- 由波动率的波动率引起
        adj2 = (params.sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
        
        # 总隐含波动率
        vol_total = vol_base + adj1 + adj2
        
        # 确保为正
        vol_total = max(vol_total, 0.01)
        
        vols.append(vol_total * 100)  # 转换为百分比
    
    return np.array(vols)


def calibrate_one_step(option_data, S0, r=0.01, q=0.0):
    """一步法校准：同时优化所有参数"""
    print("\n" + "="*70)
    print("ONE-STEP CALIBRATION: 同时优化所有参数")
    print("="*70)
    
    start_time = time.time()
    
    # 筛选有效数据
    valid_data = [opt for opt in option_data 
                  if opt['mark_iv'] > 0 and opt['T'] > 0.005]
    
    print(f"使用 {len(valid_data)} 个有效期权进行校准")
    
    def objective(params):
        kappa, theta, sigma, rho, v0 = params
        
        # Feller条件惩罚
        feller_violation = max(0.0, sigma**2 - 2.0 * kappa * theta)
        penalty = 1e5 * feller_violation
        
        errors = []
        for opt in valid_data:
            T = opt['T']
            moneyness = opt['K'] / S0
            log_m = np.log(moneyness)
            
            # 计算模型IV
            E_vT = theta + (v0 - theta) * np.exp(-kappa * T)
            vol_base = np.sqrt(max(E_vT, 1e-8))
            adj1 = rho * sigma * np.sqrt(T) * log_m / 2.0
            adj2 = (sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
            model_iv = vol_base + adj1 + adj2
            
            # 市场IV（已经是小数形式）
            market_iv = opt['mark_iv']
            
            # 相对误差
            error = ((model_iv - market_iv) / market_iv) ** 2
            errors.append(error)
        
        return np.mean(errors) + penalty if errors else 1e10
    
    # 边界 - 适合比特币的高波动率
    bounds = [
        (0.1, 10.0),    # kappa
        (0.04, 1.0),    # theta (20%-100% vol)
        (0.1, 3.0),     # sigma
        (-0.95, 0.95),  # rho
        (0.04, 1.0)     # v0 (20%-100% vol)
    ]
    
    # 多个起始点
    starting_points = [
        [2.0, 0.25, 1.0, -0.6, 0.30],   # ~50% vol
        [3.0, 0.36, 1.5, -0.7, 0.40],   # ~60% vol
    ]
    
    best_result = None
    best_error = float('inf')
    
    for i, x0 in enumerate(starting_points):
        print(f"\n  尝试起始点 {i+1}: κ={x0[0]:.2f}, θ={x0[1]:.3f}, σ={x0[2]:.2f}, ρ={x0[3]:.2f}, v₀={x0[4]:.3f}")
        
        result = minimize(
            objective, x0, bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        print(f"    目标函数值: {result.fun:.6f}")
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
            print("    ✓ 新的最优解!")
    
    params = HestonParams(
        kappa=best_result.x[0],
        theta=best_result.x[1],
        sigma=best_result.x[2],
        rho=best_result.x[3],
        v0=best_result.x[4]
    )
    
    elapsed = time.time() - start_time
    print(f"\n一步法校准完成 (耗时: {elapsed:.1f}s)")
    print("="*70)
    print(f"  κ (kappa):     {params.kappa:.4f}")
    print(f"  θ (theta):     {params.theta:.4f} -> 长期波动率: {np.sqrt(params.theta)*100:.1f}%")
    print(f"  σ (sigma):     {params.sigma:.4f}")
    print(f"  ρ (rho):       {params.rho:.4f}")
    print(f"  v₀ (v0):       {params.v0:.4f} -> 当前波动率: {np.sqrt(params.v0)*100:.1f}%")
    print(f"  目标函数值:    {best_result.fun:.6f}")
    
    return params


def calibrate_two_step(option_data, S0, r=0.01, q=0.0):
    """两步法校准"""
    print("\n" + "="*70)
    print("TWO-STEP CALIBRATION: 两阶段校准")
    print("="*70)
    
    start_time = time.time()
    
    # 筛选有效数据
    valid_data = [opt for opt in option_data 
                  if opt['mark_iv'] > 0 and opt['T'] > 0.005]
    
    print(f"使用 {len(valid_data)} 个有效期权进行校准")
    
    # 第一阶段：校准波动率参数（固定rho=-0.5）
    print("\n第一阶段：校准波动率参数...")
    
    def stage1_objective(params):
        kappa, theta, sigma, v0 = params
        rho = -0.5  # 固定
        
        feller_violation = max(0.0, sigma**2 - 2.0 * kappa * theta)
        penalty = 1e5 * feller_violation
        
        errors = []
        for opt in valid_data:
            T = opt['T']
            moneyness = opt['K'] / S0
            log_m = np.log(moneyness)
            
            E_vT = theta + (v0 - theta) * np.exp(-kappa * T)
            vol_base = np.sqrt(max(E_vT, 1e-8))
            adj1 = rho * sigma * np.sqrt(T) * log_m / 2.0
            adj2 = (sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
            model_iv = vol_base + adj1 + adj2
            
            market_iv = opt['mark_iv']
            error = ((model_iv - market_iv) / market_iv) ** 2
            errors.append(error)
        
        return np.mean(errors) + penalty if errors else 1e10
    
    bounds1 = [(0.1, 10.0), (0.04, 1.0), (0.1, 3.0), (0.04, 1.0)]
    
    result1 = differential_evolution(
        stage1_objective, bounds1,
        maxiter=100, popsize=15, tol=1e-6, seed=42
    )
    
    kappa, theta, sigma, v0 = result1.x
    print(f"  κ={kappa:.4f}, θ={theta:.4f}, σ={sigma:.4f}, v₀={v0:.4f}")
    
    # 第二阶段：校准rho
    print("\n第二阶段：校准相关系数...")
    
    def stage2_objective(rho_arr):
        rho = rho_arr[0]
        
        errors = []
        for opt in valid_data:
            T = opt['T']
            moneyness = opt['K'] / S0
            log_m = np.log(moneyness)
            
            E_vT = theta + (v0 - theta) * np.exp(-kappa * T)
            vol_base = np.sqrt(max(E_vT, 1e-8))
            adj1 = rho * sigma * np.sqrt(T) * log_m / 2.0
            adj2 = (sigma**2 * T / 8.0) * (log_m**2 - 2*log_m)
            model_iv = vol_base + adj1 + adj2
            
            market_iv = opt['mark_iv']
            error = ((model_iv - market_iv) / market_iv) ** 2
            errors.append(error)
        
        return np.mean(errors) if errors else 1e10
    
    result2 = minimize(
        stage2_objective, [-0.5],
        bounds=[(-0.95, 0.95)],
        method='L-BFGS-B'
    )
    
    rho = result2.x[0]
    
    params = HestonParams(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)
    
    elapsed = time.time() - start_time
    print(f"\n两步法校准完成 (耗时: {elapsed:.1f}s)")
    print("="*70)
    print(f"  κ (kappa):     {params.kappa:.4f}")
    print(f"  θ (theta):     {params.theta:.4f} -> 长期波动率: {np.sqrt(params.theta)*100:.1f}%")
    print(f"  σ (sigma):     {params.sigma:.4f}")
    print(f"  ρ (rho):       {params.rho:.4f}")
    print(f"  v₀ (v0):       {params.v0:.4f} -> 当前波动率: {np.sqrt(params.v0)*100:.1f}%")
    
    return params


def plot_volatility_matching_all_maturities(option_data, S0, one_step_params, two_step_params,
                                            save_path='plots/volatility_matching_fixed.png'):
    """
    绘制所有到期日的波动率匹配图
    
    关键修复：
    1. mark_iv已经是百分比形式，不需要再乘以100
    2. 正确计算模型波动率曲线
    """
    print("\n" + "="*70)
    print("生成波动率匹配图（所有到期日）")
    print("="*70)
    
    # 转换为DataFrame
    df = pd.DataFrame(option_data)
    df['moneyness'] = df['K'] / S0
    
    # 获取所有唯一到期日
    expiries = sorted(df['expiry_date'].unique(), 
                     key=lambda x: df[df['expiry_date']==x]['T'].iloc[0])
    
    n_expiries = len(expiries)
    print(f"发现 {n_expiries} 个到期日: {expiries}")
    
    # 计算子图布局
    n_cols = min(4, n_expiries)
    n_rows = (n_expiries + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
    
    # 如果只有一行或一列，确保axes是2D数组
    if n_expiries == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for idx, expiry in enumerate(expiries):
        ax = axes[idx]
        
        # 筛选当前到期日的数据
        exp_data = df[df['expiry_date'] == expiry].copy()
        T = exp_data['T'].iloc[0]
        
        print(f"\n  到期日 {idx+1}/{n_expiries}: {expiry}")
        print(f"    T = {T:.4f} years ({T*365:.1f} days)")
        print(f"    数据点: {len(exp_data)}")
        
        # 筛选有效moneyness范围
        exp_data = exp_data[(exp_data['moneyness'] > 0.6) & 
                           (exp_data['moneyness'] < 1.6) &
                           (exp_data['mark_iv'] > 0)].copy()
        
        if len(exp_data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(f'{expiry}')
            continue
        
        # 市场数据 - mark_iv已经是百分比形式（如52.08）
        # 这里直接使用，不需要乘以100
        moneyness_market = exp_data['moneyness'].values
        iv_market = exp_data['mark_iv'].values * 100  # mark_iv是小数形式（已经除以100了），转回百分比
        
        print(f"    市场IV范围: {iv_market.min():.1f}% - {iv_market.max():.1f}%")
        
        # 生成模型曲线
        moneyness_grid = np.linspace(0.6, 1.6, 200)
        
        one_vols = calculate_heston_iv(one_step_params, T, moneyness_grid)
        two_vols = calculate_heston_iv(two_step_params, T, moneyness_grid)
        
        print(f"    一步法IV范围: {one_vols.min():.1f}% - {one_vols.max():.1f}%")
        print(f"    两步法IV范围: {two_vols.min():.1f}% - {two_vols.max():.1f}%")
        
        # 绘制市场数据点（黑色圆点）
        ax.scatter(moneyness_market, iv_market,
                  s=60, c='black', alpha=0.7,
                  edgecolors='white', linewidths=0.8,
                  label='Market IV', zorder=3)
        
        # 绘制一步法模型曲线（蓝色实线）
        ax.plot(moneyness_grid, one_vols, 
               color='#1f77b4', linewidth=3,
               linestyle='-', alpha=0.9,
               label='One-Step Heston', zorder=2)
        
        # 绘制两步法模型曲线（绿色虚线）
        ax.plot(moneyness_grid, two_vols,
               color='#2ca02c', linewidth=3,
               linestyle='--', alpha=0.9,
               label='Two-Step Heston', zorder=2)
        
        # ATM参考线
        ax.axvline(1.0, color='red', linestyle=':', 
                  linewidth=2, alpha=0.6, label='ATM')
        
        # 格式设置
        ax.set_xlabel('Moneyness (K/S₀)', fontsize=10)
        ax.set_ylabel('Implied Volatility (%)', fontsize=10)
        ax.set_title(f'{expiry}\nT = {T:.4f}y ({T*365:.1f}d)', fontsize=10, fontweight='bold')
        
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 动态Y轴范围
        all_vols = np.concatenate([iv_market, one_vols, two_vols])
        y_min = max(0, np.percentile(all_vols, 1) - 5)
        y_max = np.percentile(all_vols, 99) + 5
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.6, 1.6)
    
    # 隐藏多余的子图
    for idx in range(n_expiries, len(axes)):
        axes[idx].set_visible(False)
    
    # 添加参数信息
    param_text = (
        f"One-Step: κ={one_step_params.kappa:.2f}, θ={one_step_params.theta:.3f} "
        f"(σ={np.sqrt(one_step_params.theta)*100:.0f}%), "
        f"σ_v={one_step_params.sigma:.2f}, ρ={one_step_params.rho:.2f}, "
        f"v₀={one_step_params.v0:.3f} (σ₀={np.sqrt(one_step_params.v0)*100:.0f}%)\n"
        f"Two-Step: κ={two_step_params.kappa:.2f}, θ={two_step_params.theta:.3f} "
        f"(σ={np.sqrt(two_step_params.theta)*100:.0f}%), "
        f"σ_v={two_step_params.sigma:.2f}, ρ={two_step_params.rho:.2f}, "
        f"v₀={two_step_params.v0:.3f} (σ₀={np.sqrt(two_step_params.v0)*100:.0f}%)"
    )
    
    fig.suptitle('Heston Model: Expected Volatility vs Market Implied Volatility\n' +
                 'Bitcoin Options - All Maturities',
                fontsize=14, fontweight='bold', y=1.02)
    
    fig.text(0.5, -0.02, param_text, ha='center', fontsize=9, 
            family='monospace', style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    print(f"\n{'='*70}")
    print(f"✓ 图像已保存至: {save_path}")
    print(f"{'='*70}\n")
    
    return fig


def main():
    """主函数"""
    print("="*70)
    print("HESTON模型校准对比 - 修复版")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    print("\n加载数据...")
    option_data, S0 = parse_option_data("data/deribit_btc_options_clean.csv")
    print(f"✓ 加载 {len(option_data)} 个期权 (S₀ = ${S0:.2f})")
    
    # 检查mark_iv的范围
    ivs = [opt['mark_iv'] for opt in option_data if opt['mark_iv'] > 0]
    print(f"  市场IV范围: {min(ivs)*100:.1f}% - {max(ivs)*100:.1f}%")
    
    # 一步法校准
    one_step_params = calibrate_one_step(option_data, S0)
    
    # 两步法校准
    two_step_params = calibrate_two_step(option_data, S0)
    
    # 绘制所有到期日的波动率匹配图
    plot_volatility_matching_all_maturities(
        option_data, S0, one_step_params, two_step_params
    )
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == "__main__":
    main()

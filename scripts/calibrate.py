# calibrate.py
import sys
sys.path.append('src')

from heston import HestonModel, parse_option_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. 加载数据
    option_data, S0 = parse_option_data("data/deribit_btc_options_clean.csv")
    
    # 2. 初始化模型
    model = HestonModel(kappa=1.0, theta=0.04, sigma=0.3,
                        rho=-0.5, v0=0.04, r=0.01, q=0.0)
    
    # 3. 测试单个期权定价
    first_opt = option_data[0]
    price = model.option_price(S0, first_opt["K"],
                               first_opt["T"], first_opt["type"])
    print(f"Model price for first option: {price:.4f} USD")
    
    # 4. 校准模型（使用所有期权）
    print(f"\n开始校准Heston模型 (使用 {len(option_data)} 个期权)...")
    result = model.calibrate(option_data, S0)
    
    print("\n校准成功！")
    print(f"kappa (均值回归速度): {result.x[0]:.4f}")
    print(f"theta (长期波动率): {result.x[1]:.4f}")
    print(f"sigma (波动率的波动率): {result.x[2]:.4f}")
    print(f"rho (相关系数): {result.x[3]:.4f}")
    print(f"v0 (初始波动率): {result.x[4]:.4f}")
    print(f"\n目标函数值: {result.fun:.4f}")
    
    # 5. 验证拟合效果
    print("\n拟合效果对比:")
    for i, opt in enumerate(option_data[:469]):
        model_price = model.option_price(S0, opt["K"], opt["T"], opt["type"])
        print(f"期权{i+1}: 市场价格={opt['market_price']:.2f}, "
              f"模型价格={model_price:.2f}, "
              f"误差={abs(model_price-opt['market_price']):.2f}")


if __name__ == "__main__":
    main()
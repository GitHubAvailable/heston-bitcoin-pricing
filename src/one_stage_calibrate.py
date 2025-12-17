# calibrate.py
import sys

sys.path.append('src')

from heston import HestonModel, parse_option_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. Load Data
    option_data, S0 = parse_option_data("../data/deribit_btc_options_clean.csv")

    # 2. Initialize Model
    model = HestonModel(kappa=1.0, theta=0.04, sigma=0.3,
                        rho=-0.5, v0=0.04, r=0.01, q=0.0)

    # 3. Test Single Option Pricing
    first_opt = option_data[0]
    price = model.option_price(S0, first_opt["K"],
                               first_opt["T"], first_opt["type"])
    print(f"Model price for first option: {price:.4f} USD")

    # 4. Calibrate Model (using all options)
    print(f"\nStarting Heston model calibration (using {len(option_data)} options)...")
    result = model.calibrate(option_data, S0)

    print("\nCalibration Successful!")
    print(f"kappa (Mean reversion speed): {result.x[0]:.4f}")
    print(f"theta (Long-term variance): {result.x[1]:.4f}")
    print(f"sigma (Vol of vol): {result.x[2]:.4f}")
    print(f"rho (Correlation): {result.x[3]:.4f}")
    print(f"v0 (Initial variance): {result.x[4]:.4f}")
    print(f"\nObjective function value: {result.fun:.4f}")

    # 5. Verify Fit Quality
    print("\nFit comparison:")
    for i, opt in enumerate(option_data[:469]):
        model_price = model.option_price(S0, opt["K"], opt["T"], opt["type"])
        print(f"Option {i + 1}: Market Price={opt['market_price']:.2f}, "
              f"Model Price={model_price:.2f}, "
              f"Error={abs(model_price - opt['market_price']):.2f}")


if __name__ == "__main__":
    main()
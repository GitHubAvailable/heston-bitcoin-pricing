import sys
sys.path.append('src')
from two_stage_heston_calibration import TwoStageHestonCalibrator
from heston import parse_option_data

# 1. Load Data
option_data, S0 = parse_option_data("../data/deribit_btc_options_clean.csv")

# 2. Initialize Calibrator
calibrator = TwoStageHestonCalibrator(r=0.01, q=0.0)

# 3. Stage 1: Calibrate Volatility Parameters
stage1_result = calibrator.calibrate_stage1(option_data, S0)

# 4. Stage 2: Calibrate Correlation Coefficient
stage2_result = calibrator.calibrate_stage2(option_data, S0)

# 5. Evaluate Calibration Fit
fit_stats = calibrator.evaluate_fit(option_data, S0,
                                     save_path='plots/calibration_fit.png')

# 6. Pricing using the Calibrated Model
model = calibrator.model
new_price = model.option_price(S0=95000, K=100000, T=0.5, option_type='call')
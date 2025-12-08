import sys
sys.path.append('src')
from two_stage_heston_calibration import TwoStageHestonCalibrator
from heston import parse_option_data

# 1. 加载数据
option_data, S0 = parse_option_data("data/deribit_btc_options_clean.csv")

# 2. 创建校准器
calibrator = TwoStageHestonCalibrator(r=0.01, q=0.0)

# 3. 第一阶段：校准波动率参数
stage1_result = calibrator.calibrate_stage1(option_data, S0)

# 4. 第二阶段：校准相关系数
stage2_result = calibrator.calibrate_stage2(option_data, S0)

# 5. 评估拟合效果
fit_stats = calibrator.evaluate_fit(option_data, S0, 
                                     save_path='plots/calibration_fit.png')

# 6. 使用校准后的模型定价
model = calibrator.model
new_price = model.option_price(S0=95000, K=100000, T=0.5, option_type='call')
import sys
sys.path.append('src')
from implied_volatility import *

# 1. 计算隐含波动率
df = calculate_iv_from_data('data/deribit_btc_options_clean.csv')

# 2. 查看统计信息
stats = summary_statistics(df)
print(stats)

# 3. 生成各种图表
plot_volatility_smile(df, expiry_date='7DEC25')
plot_volatility_surface(df)
plot_term_structure(df)
plot_skew_comparison(df)
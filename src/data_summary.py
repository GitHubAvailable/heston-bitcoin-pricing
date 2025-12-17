import sys
sys.path.append('src')
from implied_volatility import *

# 1. Calculate Implied Volatility
df = calculate_iv_from_data('../data/deribit_btc_options_clean.csv')

# 2. View Summary Statistics
stats = summary_statistics(df)
print(stats)

# 3. Generate Various Plots
plot_volatility_smile(df, expiry_date='7DEC25')
plot_volatility_surface(df)
plot_term_structure(df)
plot_skew_comparison(df)
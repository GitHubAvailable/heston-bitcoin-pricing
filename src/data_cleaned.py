import requests
import pandas as pd
from datetime import datetime
import os


def get_deribit_option_data(currency='BTC'):
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {
        "currency": currency,
        "kind": "option"
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching full {currency} option data from Deribit...")
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    if 'result' not in data:
        print("No results obtained. Please check network or parameters.")
        return None

    options_list = []
    now = datetime.utcnow()

    for entry in data['result']:
        instrument_name = entry['instrument_name']
        parts = instrument_name.split('-')
        if len(parts) < 4: continue

        expiry_str = parts[1]
        strike = float(parts[2])
        option_type = 'call' if parts[3] == 'C' else 'put'
        S0 = entry.get('underlying_price')

        # Calculate time to maturity T
        try:
            dt = datetime.strptime(expiry_str, "%d%b%y")
            dt = dt.replace(hour=8)
            delta = dt - now
            days = delta.days + delta.seconds / (24 * 3600)
            T = max(days / 365.0, 0.0001)
        except:
            T = 0

        # Extract key price information
        bid_btc = entry.get('bid_price') or 0.0
        ask_btc = entry.get('ask_price') or 0.0
        mark_price = entry.get('mark_price') or 0.0
        mark_iv = entry.get('mark_iv') or 0.0

        # Calculate relative spread (Spread %), for subsequent analysis (optional)
        # Large spread implies high market divergence or low liquidity
        spread_pct = (ask_btc - bid_btc) / mark_price if mark_price > 0 else 0

        item = {
            'instrument_name': instrument_name,
            'type': option_type,
            'expiry_date': expiry_str,
            'strike': strike,
            'T_years': round(T, 5),
            'S0': S0,
            'market_price_btc': mark_price,
            'market_price_usd': mark_price * S0,
            'mark_iv': mark_iv,
            'bid_btc': bid_btc,
            'ask_btc': ask_btc,
            'spread_pct': round(spread_pct, 4),
            'open_interest': entry.get('open_interest')
        }
        options_list.append(item)

    df = pd.DataFrame(options_list)
    return df


if __name__ == "__main__":
    df = get_deribit_option_data("BTC")

    if df is not None:
        print(f"Original data count: {len(df)}")

        # --- Core Data Cleaning and Filtering Logic ---

        # 1. Basic Filter: Remove short-dated data (T < 1 day is noisy, high Gamma risk, unsuitable for Heston calibration)
        # Recommendation: Retain data > 2 days, or keep original 0.002 (~17 hours)
        mask_time = df['T_years'] > 0.004  # 0.004 is approx 1.5 days

        # 2. Liquidity Filter: Must have buy orders (Bid > 0)
        # Critical step. Bid = 0 implies illiquidity and invalid market pricing.
        mask_liquidity = df['bid_btc'] > 0

        # 3. Price Validity: Market price cannot be too close to 0
        # Deep OTM options with negligible prices (e.g., 0.0001) are useless for calibration and cause division by zero errors
        mask_price = df['market_price_btc'] > 0.0005

        # 4. IV Validity: Exchange must provide valid IV
        mask_iv = df['mark_iv'] > 0

        # Apply all filters
        df_clean = df[mask_time & mask_liquidity & mask_price & mask_iv].copy()

        # --- Sorting Optimization ---
        # Sorting facilitates Volatility Smile observation:
        # Group by Maturity -> Strike (Ascending) -> Call/Put pairs
        df_clean = df_clean.sort_values(by=['T_years', 'strike', 'type'], ascending=[True, True, True])

        # Save CSV
        filename = '../data/deribit_btc_options_clean.csv'
        df_clean.to_csv(filename, index=False)

        print(f"Cleaned data count: {len(df_clean)} (filtered out {len(df) - len(df_clean)} low quality records)")
        print(f"Data saved to: {os.path.abspath(filename)}")

        # --- Data Quality Check ---
        print("\nData Overview (First 10 records):")
        print(df_clean[['expiry_date', 'strike', 'type', 'market_price_usd', 'mark_iv', 'bid_btc', 'ask_btc']].head(10))

        # Check coverage of different maturities
        maturities = df_clean['expiry_date'].unique()
        print(f"\nValid maturities captured ({len(maturities)}): {maturities}")
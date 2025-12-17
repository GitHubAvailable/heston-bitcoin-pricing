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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching all {currency} option data from Deribit...")
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
        instrument_name = entry['instrument_name']  # e.g. BTC-29MAR24-60000-C

        parts = instrument_name.split('-')
        if len(parts) < 4: continue

        expiry_str = parts[1]
        strike = float(parts[2])
        option_type = 'call' if parts[3] == 'C' else 'put'

        # Underlying price
        S0 = entry.get('underlying_price')

        # Calculate time to maturity T (annualized)
        try:
            dt = datetime.strptime(expiry_str, "%d%b%y")
            dt = dt.replace(hour=8)  # Deribit delivery time
            delta = dt - now
            days = delta.days + delta.seconds / (24 * 3600)
            T = max(days / 365.0, 0.0001)
        except:
            T = 0

        item = {
            'instrument_name': instrument_name,
            'type': option_type,
            'expiry_date': expiry_str,
            'strike': strike,
            'T_years': round(T, 5),
            'S0': S0,
            'market_price_btc': entry.get('mark_price'),
            'market_price_usd': entry.get('mark_price') * S0,  # Target Price (USD)
            'mark_iv': entry.get('mark_iv'),
            'bid_btc': entry.get('bid_price'),
            'ask_btc': entry.get('ask_price'),
            'open_interest': entry.get('open_interest')
        }
        options_list.append(item)

    df = pd.DataFrame(options_list)
    return df


if __name__ == "__main__":
    df = get_deribit_option_data("BTC")

    if df is not None:
        # 1. Filter: Remove expired data
        df_clean = df[df['T_years'] > 0.002]

        # 2. Sort: Maturity (Near->Far) -> Strike (Low->High) -> Type
        # ascending=[True, True, True] means:
        # T_years: Small to Large
        # strike: Small to Large
        df_clean = df_clean.sort_values(by=['T_years', 'strike', 'type'], ascending=[True, True, True])

        # 3. Save CSV
        filename = '../data/deribit_btc_options.csv'
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_clean.to_csv(filename, index=False)

        print(f"\nSuccess! Data saved to: {os.path.abspath(filename)}")
        print(f"Total fetched: {len(df_clean)} records")
        print("\nData Preview:")
        # Print a sample to check alignment
        print(df_clean[['instrument_name', 'type', 'strike', 'T_years', 'market_price_usd']].head(5))
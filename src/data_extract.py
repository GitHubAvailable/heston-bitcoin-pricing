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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在从 Deribit 获取 {currency} 全量期权数据...")
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
    except Exception as e:
        print(f"请求失败: {e}")
        return None

    if 'result' not in data:
        print("未获取到结果，请检查网络或参数。")
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

        # 标的价格
        S0 = entry.get('underlying_price')

        # 计算到期时间 T (年化)
        try:
            dt = datetime.strptime(expiry_str, "%d%b%y")
            dt = dt.replace(hour=8)  # Deribit 交割时间
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
            'market_price_usd': entry.get('mark_price') * S0,  # 目标价格
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
        # 1. 过滤: 去掉过期数据
        df_clean = df[df['T_years'] > 0.002]

        # 2. 排序: 到期日(近->远) -> 类型(Call->Put) -> 行权价(低->高)
        # ascending=[True, True, True] 意味着:
        # T_years: 从小到大

        # strike: 从小到大
        df_clean = df_clean.sort_values(by=['T_years', 'strike', 'type'], ascending=[True, True, True])

        # 3. 保存 CSV
        filename = 'data/deribit_btc_options.csv'
        df_clean.to_csv(filename, index=False)

        print(f"\n成功! 数据已保存至: {os.path.abspath(filename)}")
        print(f"总共获取: {len(df_clean)} 条数据")
        print("\n数据预览 (Call vs Put 排列展示):")
        # 打印一下同一个到期日下的交界处，看看是否整齐
        print(df_clean[['instrument_name', 'type', 'strike', 'T_years', 'market_price_usd']].head(5))
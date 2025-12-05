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
        instrument_name = entry['instrument_name']
        parts = instrument_name.split('-')
        if len(parts) < 4: continue

        expiry_str = parts[1]
        strike = float(parts[2])
        option_type = 'call' if parts[3] == 'C' else 'put'
        S0 = entry.get('underlying_price')

        # 计算到期时间 T
        try:
            dt = datetime.strptime(expiry_str, "%d%b%y")
            dt = dt.replace(hour=8)
            delta = dt - now
            days = delta.days + delta.seconds / (24 * 3600)
            T = max(days / 365.0, 0.0001)
        except:
            T = 0

        # 提取关键价格信息
        bid_btc = entry.get('bid_price') or 0.0
        ask_btc = entry.get('ask_price') or 0.0
        mark_price = entry.get('mark_price') or 0.0
        mark_iv = entry.get('mark_iv') or 0.0

        # 计算相对价差 (Spread %), 用于后续分析 (可选)
        # 如果价差过大，说明市场分歧大或流动性差
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
        print(f"原始数据条数: {len(df)}")

        # --- 数据清洗与过滤核心逻辑 ---

        # 1. 基础过滤: 去掉过期数据 (T < 1天 通常噪音很大，Gamma风险极高，不适合做Heston校准)
        # 建议保留至少 2 天以上的数据，或者保持你原来的 0.002 (约17小时)
        mask_time = df['T_years'] > 0.004  # 0.004 约为 1.5 天

        # 2. 流动性过滤: 必须有买单 (Bid > 0)
        # 这是最关键的一步。如果Bid为0，说明你卖不出去，市场定价失效。
        mask_liquidity = df['bid_btc'] > 0

        # 3. 价格有效性: 市场价格不能太接近0
        # 深度虚值的垃圾期权价格可能为 0.0001，这种数据对校准不仅没用，还会导致除以零错误
        mask_price = df['market_price_btc'] > 0.0005

        # 4. IV 有效性: 交易所必须能算出有效的 IV
        mask_iv = df['mark_iv'] > 0

        # 应用所有过滤器
        df_clean = df[mask_time & mask_liquidity & mask_price & mask_iv].copy()

        # --- 排序优化 ---
        # 这种排序方式方便你观察 Smile：
        # 先看同一个到期日 -> 再看Strike从小到大 -> 每一对Strike你会看到 Call 和 Put 在一起
        df_clean = df_clean.sort_values(by=['T_years', 'strike', 'type'], ascending=[True, True, True])

        # 保存 CSV
        filename = 'data/deribit_btc_options_clean.csv'
        df_clean.to_csv(filename, index=False)

        print(f"清洗后数据条数: {len(df_clean)} (过滤掉了 {len(df) - len(df_clean)} 条低质量数据)")
        print(f"数据已保存至: {os.path.abspath(filename)}")

        # --- 打印数据质量检查 ---
        print("\n数据概览 (前10条):")
        print(df_clean[['expiry_date', 'strike', 'type', 'market_price_usd', 'mark_iv', 'bid_btc', 'ask_btc']].head(10))

        # 检查一下覆盖了多少个不同的到期日
        maturities = df_clean['expiry_date'].unique()
        print(f"\n捕获到的有效到期日 ({len(maturities)}个): {maturities}")
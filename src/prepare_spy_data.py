# -*- coding: utf-8 -*-
"""
prepare_spy_data.py - Convert SPY option data to Heston model format

Converts raw SPY option data format to the format required by heston.py
Input format: Wide format with columns like [C_BID], [C_ASK], [P_BID], [P_ASK]
Output format: Long format with one row per option (strike, T_years, market_price_usd, type, etc.)
"""

import pandas as pd
import numpy as np


def convert_spy_to_heston_format(input_csv: str, output_csv: str):
    """
    Convert SPY option data to Heston model format.

    Performs minimal processing - only format conversion:
    - Cleans column names (removes brackets)
    - Calculates time to maturity in years
    - Converts wide format (one row per strike) to long format (one row per option)
    - Computes mid prices from bid/ask
    - Skips options with missing bid or ask prices

    Parameters:
    -----------
    input_csv : str
        Path to raw SPY data CSV file
    output_csv : str
        Path to save converted data

    Returns:
    --------
    pd.DataFrame
        Converted dataframe ready for Heston model calibration
    """

    print("Loading SPY option data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} strikes")

    # Clean column names: remove brackets and whitespace
    df.columns = df.columns.str.strip().str.replace('[', '').str.replace(']', '')

    # Calculate time to maturity in years
    df['T_years'] = df['DTE'] / 365.0

    # Extract spot price
    S0 = df['UNDERLYING_LAST'].iloc[0]
    df['S0'] = S0

    # Convert from wide to long format
    call_data = []
    put_data = []

    for idx, row in df.iterrows():
        # Process CALL options
        if pd.notna(row['C_BID']) and pd.notna(row['C_ASK']):
            c_mid = (row['C_BID'] + row['C_ASK']) / 2.0
            c_spread_pct = (row['C_ASK'] - row['C_BID']) / c_mid * 100.0 if c_mid > 0 else np.nan

            call_data.append({
                'strike': row['STRIKE'],
                'T_years': row['T_years'],
                'market_price_usd': c_mid,
                'type': 'call',
                'S0': S0,
                'spread_pct': c_spread_pct,
                'mark_iv': row['C_IV'] if pd.notna(row['C_IV']) else np.nan,
                'open_interest': row['C_VOLUME'] if pd.notna(row['C_VOLUME']) else 0,
                'expiry_date': row['EXPIRE_DATE'],
            })

        # Process PUT options
        if pd.notna(row['P_BID']) and pd.notna(row['P_ASK']):
            p_mid = (row['P_BID'] + row['P_ASK']) / 2.0
            p_spread_pct = (row['P_ASK'] - row['P_BID']) / p_mid * 100.0 if p_mid > 0 else np.nan

            put_data.append({
                'strike': row['STRIKE'],
                'T_years': row['T_years'],
                'market_price_usd': p_mid,
                'type': 'put',
                'S0': S0,
                'spread_pct': p_spread_pct,
                'mark_iv': row['P_IV'] if pd.notna(row['P_IV']) else np.nan,
                'open_interest': row['P_VOLUME'] if pd.notna(row['P_VOLUME']) else 0,
                'expiry_date': row['EXPIRE_DATE'],
            })

    # Combine calls and puts
    options_df = pd.DataFrame(call_data + put_data)

    # Sort by maturity then strike
    options_df = options_df.sort_values(['T_years', 'strike']).reset_index(drop=True)

    # Save to CSV
    options_df.to_csv(output_csv, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total options converted: {len(options_df)}")
    print(f"  - Calls: {len(call_data)}")
    print(f"  - Puts: {len(put_data)}")
    print(f"\nSpot price: ${S0:.2f}")
    print(f"Expiries: {options_df['expiry_date'].nunique()}")
    print(f"Maturity range: {options_df['T_years'].min():.3f} - {options_df['T_years'].max():.3f} years")
    print(f"Strike range: ${options_df['strike'].min():.0f} - ${options_df['strike'].max():.0f}")
    print(f"\nOutput saved to: {output_csv}")

    return options_df


def main():
    """Main execution function."""

    # File paths
    input_file = "../data/spy_20221230.csv"
    output_file = "../data/spy_options_clean.csv"

    # Convert data
    df = convert_spy_to_heston_format(input_file, output_file)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("Data conversion complete. You can now:")
    print("1. Use simple calibration (heston.py)")
    print("2. Use two-stage calibration (two_stage_heston_calibration.py)")
    print("\nMake sure to set r (risk-free rate) and q (dividend yield)")
    print("when initializing the Heston model.")


if __name__ == "__main__":
    main()
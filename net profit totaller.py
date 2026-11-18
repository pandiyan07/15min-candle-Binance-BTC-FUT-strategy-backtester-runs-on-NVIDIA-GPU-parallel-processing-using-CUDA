import pandas as pd

dec_csv_filename = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\15min ohlc 2024 DECEMBER.csv"
jan_csv_filename = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\15min ohlc 2025 JANUARY.csv"
feb_csv_filename = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\15min ohlc 2025 FEBRUARY.csv"
mar_csv_filename = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\15min ohlc 2025 MARCH.csv"
apr_csv_filename = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\15min ohlc 2025 APRIL.csv"

OUTPUT_FILE_NET_PROFIT = r"D:\DOWNLOADS\backtest output\15min backtest op\15min v27\v27 Net Profit total 5 months 15min backtesting.csv"

# Function 2: Load DataFrame from CSV and print
def LOAD_FROM_CSV(filename):
    loaded_df = pd.read_csv(filename)
    loaded_df = loaded_df.set_index('Long Profit')
    return loaded_df

dec_df = LOAD_FROM_CSV(dec_csv_filename)
jan_df = LOAD_FROM_CSV(jan_csv_filename)
feb_df = LOAD_FROM_CSV(feb_csv_filename)
mar_df = LOAD_FROM_CSV(mar_csv_filename)
apr_df = LOAD_FROM_CSV(apr_csv_filename)

all_df = pd.concat(
    [dec_df, jan_df, feb_df, mar_df, apr_df],
    ignore_index=True
)

# then group and sum as before
output_df = (
    all_df
    .groupby(
        ['MAX_VOL', 'MIN_VOL', 'SHORT_CANDLE_HEIGHT', 'LONG_CANDLE_HEIGHT'],
        as_index=False
    )['Net Profit']
    .sum()
    .sort_values('Net Profit', ascending=False)
)

output_df = output_df.sort_values('Net Profit', ascending=False)
output_df.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
# output_df now has one row per unique (MAX_VOL, MIN_VOL, SHORT_CANDLE_HEIGHT, LONG_CANDLE_HEIGHT)
# with Net Profit = sum of profits from all four input dfs
print(output_df)
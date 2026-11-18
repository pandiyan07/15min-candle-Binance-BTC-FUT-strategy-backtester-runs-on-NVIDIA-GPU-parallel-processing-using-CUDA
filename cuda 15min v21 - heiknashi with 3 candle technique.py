import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from datetime import datetime
from joblib import Memory

# Enable caching
memory = Memory("./cache", verbose=0)

def CALCULATE_HEIKINASHI(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = 0.0
    ha_df['ha_high'] = 0.0
    ha_df['ha_low'] = 0.0

    ha_df.iloc[0, ha_df.columns.get_loc('ha_open')] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    ha_df.iloc[0, ha_df.columns.get_loc('ha_high')] = max(df['high'].iloc[0], ha_df['ha_open'].iloc[0], ha_df['ha_close'].iloc[0])
    ha_df.iloc[0, ha_df.columns.get_loc('ha_low')] = min(df['low'].iloc[0], ha_df['ha_open'].iloc[0], ha_df['ha_close'].iloc[0])

    for i in range(1, len(df)):
        ha_open_prev = ha_df.at[i-1, 'ha_open']
        ha_close_prev = ha_df.at[i-1, 'ha_close']
        ha_open = (ha_open_prev + ha_close_prev) / 2
        ha_close = (df.at[i, 'open'] + df.at[i, 'high'] + df.at[i, 'low'] + df.at[i, 'close']) / 4
        ha_high = max(df.at[i, 'high'], ha_open, ha_close)
        ha_low = min(df.at[i, 'low'], ha_open, ha_close)
        ha_df.at[i, 'ha_open'] = ha_open
        ha_df.at[i, 'ha_close'] = ha_close
        ha_df.at[i, 'ha_high'] = ha_high
        ha_df.at[i, 'ha_low'] = ha_low

    return ha_df[['ha_open', 'ha_high', 'ha_low', 'ha_close']]

def ATR_CALCULATOR(df, length):
    return ta.atr(df['ha_high'], df['ha_low'], df['ha_close'], length=length)

def CALCULATE_SMA(df, length):
    df['oc2'] = (df['ha_open'] + df['ha_close']) / 2
    return ta.sma(df['oc2'], length=length)

# CUDA kernel with your new entry/exit logic
BACKTEST_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void BACKTEST_KERNEL(
    const float* atr,
    const float* close_price,
    const float* open_price,
    const float* sma_7,
    const float* params,
    float* results,
    const int num_candles,
    const int num_params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_params) return;

    float ATR_value = params[tid];
    int result_idx = tid * 12;

    float long_total_profit = 0.0f;
    float long_total_loss = 0.0f;
    float short_total_profit = 0.0f;
    float short_total_loss = 0.0f;
    int num_long_trades = 0;
    int num_short_trades = 0;
    float entry_price = 0.0f;
    float position = 0.0f;

    for (int j = 2; j < num_candles; j++) {
        float close = close_price[j];
        float atr_val = atr[j];
        float curr_open = open_price[j];
        float curr_close = close_price[j];
        float prev_open = open_price[j - 1];
        float prev_close = close_price[j - 1];
        float sec_prev_open = open_price[j - 2];
        float sec_prev_close = close_price[j - 2];

        float curr_oc_diff = (curr_close > curr_open)
            ? ((curr_close - curr_open) * 0.8f) + curr_open
            : curr_open - ((curr_open - curr_close) * 0.8f);

        float prev_oc_diff = (prev_close > prev_open)
            ? ((prev_close - prev_open) * 0.8f) + prev_open
            : prev_open - ((prev_open - prev_close) * 0.8f);

        bool long_condition = ((sma_7[j] - sma_7[j-2]) > 13.0f) &&
                            (atr_val > ATR_value) &&
                            (curr_oc_diff > prev_close) &&
                            (prev_oc_diff > sec_prev_close);

        bool short_condition = ((sma_7[j-2] - sma_7[j]) > 13.0f) &&
                            (atr_val > ATR_value) &&
                            (curr_oc_diff < prev_close) &&
                            (prev_oc_diff < sec_prev_close);

        if (position == 0.0f) {
            if (long_condition) {
                position = 1.0f;
                entry_price = close;
                num_long_trades++;
            } else if (short_condition) {
                position = 2.0f;
                entry_price = close;
                num_short_trades++;
            }
        } else if (position == 1.0f && sma_7[j]) {
            float profit = close - entry_price;
            if (profit > 0.0f) long_total_profit += profit;
            else long_total_loss += profit;
            position = 0.0f;
        } else if (position == 2.0f && sma_7[j]) {
            float profit = entry_price - close;
            if (profit > 0.0f) short_total_profit += profit;
            else short_total_loss += profit;
            position = 0.0f;
        }

        if (j == num_candles - 1 && position != 0.0f) {
            float profit = (position == 1.0f) ? (close - entry_price) : (entry_price - close);
            if (profit > 0.0f)
                (position == 1.0f ? long_total_profit : short_total_profit) += profit;
            else
                (position == 1.0f ? long_total_loss : short_total_loss) += profit;
            position = 0.0f;
        }
    }

    float total_profit = long_total_profit + short_total_profit;
    float total_loss = fabsf(long_total_loss + short_total_loss);
    float win_rate = (num_long_trades + num_short_trades > 0)
        ? (total_profit / (total_profit + total_loss)) * 100.0f : 0.0f;
    float long_loss_pct = (long_total_profit > fabsf(long_total_loss))
        ? (fabsf(long_total_loss) / long_total_profit) * 100.0f : 0.0f;
    float short_loss_pct = (short_total_profit > fabsf(short_total_loss))
        ? (fabsf(short_total_loss) / short_total_profit) * 100.0f : 0.0f;

    results[result_idx] = ATR_value;
    results[result_idx + 1] = num_long_trades + num_short_trades;
    results[result_idx + 2] = num_long_trades;
    results[result_idx + 3] = num_short_trades;
    results[result_idx + 4] = win_rate;
    results[result_idx + 5] = long_total_profit;
    results[result_idx + 6] = long_total_loss;
    results[result_idx + 7] = long_loss_pct;
    results[result_idx + 8] = short_total_profit;
    results[result_idx + 9] = short_total_loss;
    results[result_idx + 10] = short_loss_pct;
    results[result_idx + 11] = total_profit - total_loss;
}
''', 'BACKTEST_KERNEL')

def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params = len(params)
    results = cp.zeros((num_params * 12), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    gpu_data = {
        'atr': cp.asarray(df['atr'], dtype=cp.float32),
        'close': cp.asarray(df['ha_close'], dtype=cp.float32),
        'open': cp.asarray(df['ha_open'], dtype=cp.float32),
        'sma_7': cp.asarray(df['sma_7'], dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }

    BACKTEST_KERNEL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['atr'],
        gpu_data['close'],
        gpu_data['open'],
        gpu_data['sma_7'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))

    return results.get().reshape((num_params, 12))

def TRADING_BOT(df):
    ha_df = CALCULATE_HEIKINASHI(df)
    ha_df['sma_7'] = CALCULATE_SMA(ha_df, 7)
    ha_df['atr'] = ATR_CALCULATOR(ha_df, 14)
    
    params = np.array([100, 200, 300, 400, 500, 600, 800, 900, 1000], dtype=np.float32)

    start = datetime.now()
    results = BACKTEST_GPU(ha_df, params)
    end = datetime.now()
    
    print(f"Backtest completed in {(end - start).total_seconds()} seconds")

    df_results = pd.DataFrame(results, columns=[
        'ATR Parameter',
        'Total Trades',
        'Long Trades',
        'Short Trades',
        'Win Rate (%)',
        'Long Profit',
        'Long Loss',
        'Long Loss %',
        'Short Profit',
        'Short Loss',
        'Short Loss %',
        'Net Profit'
    ])
    return df_results

if __name__ == "__main__":
    INPUT_FILE = r"5min ohlc MARCH.csv"
    OUTPUT_FILE_NET_PROFIT = r"5min ohlc MARCH v8.csv"
    
    df = pd.read_csv(INPUT_FILE)
    output_df = TRADING_BOT(df)
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df_net_profit = output_df.sort_values('l', ascending=False)
    output_df_net_profit.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    
    table_generation_end_time = datetime.now()
    print('\nTop Highest net profit =')
    print(output_df_net_profit)
    print('\nTrading Bot Ended\n')



import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for technical indicator calculations
memory = Memory("./cache", verbose=0)

def ATR_CALCULATOR(df, length):
    return ta.atr(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), length=length)


def CALCULATE_EMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.ema(df['oc2'], length=length)


def CALCULATE_SMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.sma(df['oc2'], length=length)

# GPU Kernel for Backtesting with modified 3-candle entry technique
BACKTEST_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void BACKTEST_KERNEL(
    const float* atr_val,
    const float* close_price,
    const float* open_price,
    const float* ema_5,
    const float* sma_13,
    const float* params,
    float* results,
    const int num_candles,
    const int num_params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_params) return;

    // Parameter extraction (3 params per tid)
    float sma13_LONG_ENTRY_variation = params[tid * 3];
    float sma13_SHORT_ENTRY_variation = params[tid * 3 + 1];
    float ATR_value = params[tid * 3 + 2];

    int result_idx = tid * 14;
    float long_total_profit = 0.0f;
    float long_total_loss = 0.0f;
    float short_total_profit = 0.0f;
    float short_total_loss = 0.0f;
    int num_long_trades = 0;
    int num_short_trades = 0;
    float position = 0.0f;
    float entry_price = 0.0f;

    for (int j = 2; j < num_candles; j++) {
        float close = close_price[j];
        float atr = atr_val[j];

        float curr_close = close_price[j];
        float curr_open  = open_price[j];
        float prev_close = close_price[j-1];
        float prev_open  = open_price[j-1];
        float sec_prev_close = close_price[j-2];
        float sec_prev_open  = open_price[j-2];

        // Calculate open-close difference with 80% weight
        float curr_oc_diff = (curr_close > curr_open)
            ? ((curr_close - curr_open) * 0.8f) + curr_open
            : curr_open - ((curr_open - curr_close) * 0.8f);
        float prev_oc_diff = (prev_close > prev_open)
            ? ((prev_close - prev_open) * 0.8f) + prev_open
            : prev_open - ((prev_open - prev_close) * 0.8f);

        bool long_condition = (ema_5[j] > sma_13[j]) &&
                              ((sma_13[j] - sma_13[j-2]) > sma13_LONG_ENTRY_variation) &&
                              (atr > ATR_value) &&
                              (curr_oc_diff > prev_close) &&
                              (prev_oc_diff > sec_prev_close);

        bool short_condition = (ema_5[j] < sma_13[j]) &&
                               ((sma_13[j-2] - sma_13[j]) > sma13_SHORT_ENTRY_variation) &&
                               (atr > ATR_value) &&
                               (curr_oc_diff < prev_close) &&
                               (prev_oc_diff < sec_prev_close);

        // Entry logic
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

        // Exit logic for long
        } else if (position == 1.0f) {
            if (ema_5[j] < sma_13[j]) {
                float profit = close - entry_price;
                if (profit > 0.0f) long_total_profit += profit;
                else long_total_loss += profit;
                position = 0.0f;
            }

        // Exit logic for short
        } else if (position == 2.0f) {
            if (ema_5[j] > sma_13[j]) {
                float profit = entry_price - close;
                if (profit > 0.0f) short_total_profit += profit;
                else short_total_loss += profit;
                position = 0.0f;
            }
        }

        // Force exit on last candle
        if (j == num_candles - 1 && position != 0.0f) {
            float profit = (position == 1.0f) ? (close - entry_price) : (entry_price - close);
            if (profit > 0.0f) {
                if (position == 1.0f) long_total_profit += profit;
                else short_total_profit += profit;
            } else {
                if (position == 1.0f) long_total_loss += profit;
                else short_total_loss += profit;
            }
            position = 0.0f;
        }
    }

    // Compile results
    float total_profit = long_total_profit + short_total_profit;
    float total_loss   = fabsf(long_total_loss + short_total_loss);
    float win_rate = (num_long_trades + num_short_trades > 0)
                     ? (total_profit / (total_profit + total_loss)) * 100.0f : 0.0f;
    float long_loss_pct = (long_total_profit > fabsf(long_total_loss))
                         ? (fabsf(long_total_loss) / long_total_profit) * 100.0f : 0.0f;
    float short_loss_pct= (short_total_profit > fabsf(short_total_loss))
                         ? (fabsf(short_total_loss) / short_total_profit) * 100.0f : 0.0f;

    results[result_idx    ] = sma13_LONG_ENTRY_variation;
    results[result_idx + 1] = sma13_SHORT_ENTRY_variation;
    results[result_idx + 2] = ATR_value;
    results[result_idx + 3] = num_long_trades + num_short_trades;
    results[result_idx + 4] = num_long_trades;
    results[result_idx + 5] = num_short_trades;
    results[result_idx + 6] = win_rate;
    results[result_idx + 7] = long_total_profit;
    results[result_idx + 8] = long_total_loss;
    results[result_idx + 9] = long_loss_pct;
    results[result_idx + 10] = short_total_profit;
    results[result_idx + 11] = short_total_loss;
    results[result_idx + 12] = short_loss_pct;
    results[result_idx + 13] = total_profit - total_loss;
}
''', 'BACKTEST_KERNEL')

def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params  = len(params)
    results     = cp.zeros((num_params * 14), dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid   = (num_params + threads_per_block - 1) // threads_per_block

    # Prepare GPU arrays
    gpu_data = {
        'atr':        cp.asarray(df['atr'],    dtype=cp.float32),
        'close':      cp.asarray(df['close'],  dtype=cp.float32),
        'open_price': cp.asarray(df['open'],   dtype=cp.float32),
        'ema_5':      cp.asarray(df['ema_5'],  dtype=cp.float32),
        'sma_13':     cp.asarray(df['sma_13'], dtype=cp.float32),
        'params':     cp.asarray(params,       dtype=cp.float32)
    }

    BACKTEST_KERNEL(
        (blocks_per_grid,), (threads_per_block,), (
            gpu_data['atr'],
            gpu_data['close'],
            gpu_data['open_price'],
            gpu_data['ema_5'],
            gpu_data['sma_13'],
            gpu_data['params'],
            results,
            num_candles,
            num_params
        )
    )
    return results.get().reshape((num_params, 14))


def TRADING_BOT(df):
    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # Calculate indicators
    df['ema_5']  = CALCULATE_EMA(df, 5)
    df['sma_13'] = CALCULATE_SMA(df, 13)
    df['atr'] = ATR_CALCULATOR(df, 14)

    # Drop rows with NaNs from indicator warm-up
    df = df.dropna().reset_index(drop=True)

    # Parameter grid: (long_var, short_var, ATR_threshold)
    params = np.array([
        (long_var, short_var, ATR_val)
        for long_var  in range(20, 300, 20)
        for short_var in range(20, 300, 20)
        for ATR_val    in range(100, 500, 100)
    ], dtype=np.float32)

    start_time = datetime.now()
    results    = BACKTEST_GPU(df, params)
    end_time   = datetime.now()
    print(f'Backtesting completed in {end_time - start_time}')

    # Build result DataFrame
    df_results = pd.DataFrame(results, columns=[
        'SMA13 Long ENTRY variation',
        'SMA13 Short ENTRY variation',
        'ATR',
        'Total Trades',
        'Long Trades',
        'Short Trades',
        'Win %',
        'Long Total Profit',
        'Long Total Loss',
        'Long Loss perc from Profit',
        'Short Total Profit',
        'Short Total Loss',
        'Short Loss perc from Profit',
        'Net Profit',
    ])
    return df_results

if __name__ == "__main__":
    INPUT_FILE  = r"5min ohlc MARCH.csv"
    OUTPUT_FILE = r"5min ohlc MARCH v8.csv"

    df = pd.read_csv(INPUT_FILE)
    output_df = TRADING_BOT(df)

    # Sort by Net Profit descending
    output_df_sorted = output_df.sort_values('Net Profit', ascending=False)
    output_df_sorted.to_csv(OUTPUT_FILE, index=False)

    print("\nTop Highest Net Profit:\n", output_df_sorted.head())
    print("\nTrading Bot Completed\n")

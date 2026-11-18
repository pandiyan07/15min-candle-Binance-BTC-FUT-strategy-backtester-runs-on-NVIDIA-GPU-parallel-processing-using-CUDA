
import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for technical indicator calculations
memory = Memory("./cache", verbose=0)

def CALCULATE_EMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.ema(df['oc2'], length=length)

def CALCULATE_SMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.sma(df['oc2'], length=length)

# GPU Kernel for Backtesting with simultaneous long and short trades allowed
BACKTEST_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNEL(
        const float* close_price,
        const float* ema_5, const float* sma_34,
        const float* params,
        float* results, 
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        float long_MA_deviation = params[tid * 6];
        float short_MA_deviation = params[tid * 6 + 1];
        float sma34_LONG_ENTRY_variation = params[tid * 6 + 2];
        float sma34_SHORT_ENTRY_variation = params[tid * 6 + 3];
        float TSL_Long = params[tid * 6 + 4];
        float TSL_Short = params[tid * 6 + 5];

        int result_idx = tid * 17;
        float long_total_profit = 0.0;
        float long_total_loss = 0.0;
        float short_total_profit = 0.0;
        float short_total_loss = 0.0;
        int num_long_trades = 0;
        int num_short_trades = 0;
        
        // Separate state for long and short trades.
        bool long_active = false;
        bool short_active = false;
        
        float long_entry_price = 0.0;
        float long_highest_price = 0.0;
        
        float short_entry_price = 0.0;
        float short_lowest_price = 0.0;

        for (int j = 2; j < num_candles; j++) {
            float close = close_price[j];

            bool long_condition = (ema_5[j] > sma_34[j]) && 
                (((ema_5[j] - sma_34[j]) > long_MA_deviation) || ((sma_34[j] - sma_34[j-5]) > sma34_LONG_ENTRY_variation));
            bool short_condition = (ema_5[j] < sma_34[j]) && 
                (((sma_34[j] - ema_5[j]) > short_MA_deviation) || ((sma_34[j-5] - sma_34[j]) > sma34_SHORT_ENTRY_variation));

            // Open a new long trade if the long condition is satisfied and no long is active.
            if (!long_active && long_condition) {
                long_active = true;
                long_entry_price = close;
                long_highest_price = ema_5[j];  // Initialize the highest price tracker for trailing stop
                num_long_trades++;
            }

            // Open a new short trade if the short condition is satisfied and no short is active.
            if (!short_active && short_condition) {
                short_active = true;
                short_entry_price = close;
                short_lowest_price = ema_5[j];  // Initialize the lowest price tracker for trailing stop
                num_short_trades++;
            }

            // Check exit conditions for the active long trade.
            if (long_active) {
                long_highest_price = fmaxf(long_highest_price, ema_5[j]);  // Update highest price for trailing stop
                bool long_ema_exit = ema_5[j] < sma_34[j];
                bool long_trailing_stop = ema_5[j] < (long_highest_price - TSL_Long);
                
                if (long_ema_exit || long_trailing_stop) {
                    float profit = close - long_entry_price;
                    if (profit > 0.0) {
                        long_total_profit += profit;
                    } else {
                        long_total_loss += profit;
                    }
                    long_active = false;
                }
            }

            // Check exit conditions for the active short trade.
            if (short_active) {
                short_lowest_price = fminf(short_lowest_price, ema_5[j]);  // Update lowest price for trailing stop
                bool short_ema_exit = ema_5[j] > sma_34[j];
                bool short_trailing_stop = ema_5[j] > (short_lowest_price + TSL_Short);
                
                if (short_ema_exit || short_trailing_stop) {
                    float profit = short_entry_price - close;
                    if (profit > 0.0) {
                        short_total_profit += profit;
                    } else {
                        short_total_loss += profit;
                    }
                    short_active = false;
                }
            }

            // Force close any open trade at the very last candle
            if (j == num_candles - 1) {  
                if (long_active) {
                    float profit = close - long_entry_price;
                    if (profit > 0.0) {
                        long_total_profit += profit;
                    } else {
                        long_total_loss += profit;
                    }
                    long_active = false;
                }
                if (short_active) {
                    float profit = short_entry_price - close;
                    if (profit > 0.0) {
                        short_total_profit += profit;
                    } else {
                        short_total_loss += profit;
                    }
                    short_active = false;
                }
            }
        }

        results[result_idx] = long_MA_deviation;
        results[result_idx + 1] = sma34_LONG_ENTRY_variation;
        results[result_idx + 2] = TSL_Long; 
        results[result_idx + 3] = num_long_trades;
        results[result_idx + 4] = long_total_profit;
        results[result_idx + 5] = long_total_loss;
        if (long_total_profit > fabsf(long_total_loss)) {
            results[result_idx + 6] = (fabsf(long_total_loss) / long_total_profit) * 100.0f;
        } else {
            results[result_idx + 6] = 0.0f;
        }
        
        results[result_idx + 7] = short_MA_deviation;
        results[result_idx + 8] = sma34_SHORT_ENTRY_variation; 
        results[result_idx + 9] = TSL_Short;
        results[result_idx + 10] = num_short_trades;
        results[result_idx + 11] = short_total_profit;
        results[result_idx + 12] = short_total_loss;
        if (short_total_profit > fabsf(short_total_loss)) {
            results[result_idx + 13] = (fabsf(short_total_loss) / short_total_profit) * 100.0f;
        } else {
            results[result_idx + 13] = 0.0f;
        }
        
        results[result_idx + 14] = num_long_trades + num_short_trades;
        results[result_idx + 15] = ((num_long_trades + num_short_trades) > 0) ? 
            ((long_total_profit + short_total_profit) / ((long_total_profit + short_total_profit) + fabsf(long_total_loss + short_total_loss))) * 100.0f : 0.0f;
        results[result_idx + 16] = (long_total_profit + short_total_profit) - fabsf(long_total_loss + short_total_loss); 
    }
    ''',
    'BACKTEST_KERNEL'
)


def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params = len(params)
    results = cp.zeros((num_params * 17), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    gpu_data = {
        'close': cp.asarray(df['close'], dtype=cp.float32),
        'ema_5': cp.asarray(df['ema_5'], dtype=cp.float32),
        'sma_34': cp.asarray(df['sma_34'], dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }
    
    BACKTEST_KERNEL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['sma_34'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))
    return results.get().reshape((num_params, 17))

def TRADING_BOT(df):
    df['ema_5'] = CALCULATE_EMA(df, 5)
    df['sma_34'] = CALCULATE_SMA(df, 34)
    
    params = np.array([
        (long_MA_deviation, short_MA_deviation, sma34_LONG_ENTRY_variation, sma34_SHORT_ENTRY_variation, TSL_Long, TSL_Short)
        for long_MA_deviation in range(50, 450, 50)
        for short_MA_deviation in range(50, 450, 50)
        for sma34_LONG_ENTRY_variation in range(50, 450, 50)
        for sma34_SHORT_ENTRY_variation in range(50, 450, 50)
        for TSL_Long in range(100, 600, 100)
        for TSL_Short in range(100, 600, 100)
    ], dtype=np.float32)
    
    start_time = datetime.now()
    results = BACKTEST_GPU(df, params)
    end_time = datetime.now()
    
    print(f'Backtesting completed in {end_time - start_time} seconds')
    df_results = pd.DataFrame(results, columns=[
        'Long EMA Deviation', 
        'SMA34 Long Variation', 
        'TSL Long', 
        'Long Trades', 
        'Long Total Profit', 
        'Long Total Loss',
        'Long Loss perc from Profit',
        'Short EMA Deviation', 
        'SMA34 Short Variation',
        'TSL Short',
        'Short Trades', 
        'Short Total Profit', 
        'Short Total Loss', 
        'Short Loss perc from Profit',
        'Total Trades', 
        'Win %', 
        'Net Profit',
        ])
    return df_results

if __name__ == "__main__":
    INPUT_FILE = r"5min ohlc MARCH.csv"
    OUTPUT_FILE_NET_PROFIT = r"5min ohlc MARCH v8.csv"
    
    df = pd.read_csv(INPUT_FILE)
    output_df = TRADING_BOT(df)
    
    # Sort by net profit (descending) to see the best-performing parameter combinations.
    output_df_net_profit = output_df.sort_values('Net Profit', ascending=False)
    output_df_net_profit.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    
    table_generation_end_time = datetime.now()
    print('\nTop Highest net profit =')
    print(output_df_net_profit)
    print('\nTrading Bot Ended\n')
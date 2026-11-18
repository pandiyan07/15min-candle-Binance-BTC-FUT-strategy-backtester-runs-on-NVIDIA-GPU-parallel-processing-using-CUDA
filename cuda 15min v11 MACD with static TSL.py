'''
l1 = df['macd_line'] > df['signal_line']
l2 = df['histogram'] > long_HISTOGRAM
l3 = df['atr'] > ATR_value

long_entry_condition = l1 and l2 and l3

s1 = df['macd_line'] < df['signal_line']
s2 = df['histogram'] < short_HISTOGRAM
s3 = df['atr'] > ATR_value

short_entry_condition = s1 and s2 and s3

long_exit_condition = (df['macd_line'] < df['signal_line'] and df['sma_34'] > df['ema_5']) 
                                        OR current_prfit > Long_TSL
short_exit_condition = (df['macd_line'] > df['signal_line'] and df['sma_34'] < df['ema_5']) 
                                        OR current_prfit > Short_TSL
'''

import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for technical indicator calculations
memory = Memory("./cache", verbose=0)

def CALCULATE_MACD(df, fa, sl, sig):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    macd_df = ta.macd(df['oc2'], fast=fa, slow=sl, signal=sig)
    macd_df.rename(columns={
        f'MACD_5_34_9': 'macd_line',
        f'MACDs_5_34_9': 'signal_line',
        f'MACDh_5_34_9': 'histogram'
    }, inplace=True)
    df = df.join(macd_df)
    return df

def CALCULATE_EMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.ema(df['oc2'], length=length)

def CALCULATE_SMA(df, length):
    df['oc2'] = (df['open'].astype(float) + df['close'].astype(float)) / 2
    return ta.sma(df['oc2'], length=length)

def ATR_CALCULATOR(df,length):
    return ta.atr(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), length=length) 


# GPU Kernel for Backtesting with only one trade active at a time.
BACKTEST_KERNEL = cp.RawKernel(
    r'''
    extern "C" __global__
    void BACKTEST_KERNEL(
        const float* atr_val, const float* close_price,
        const float* ema_5, const float* sma_34,
        const float* macd_line, const float* signal_line,
        const float* histogram, const float* params,
        float* results, 
        const int num_candles,
        const int num_params)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_params) return;

        float long_HISTOGRAM = params[tid * 5];
        float short_HISTOGRAM = params[tid * 5 + 1];
        float ATR_value = params[tid * 5 + 2];
        float Long_TSL = params[tid * 5 + 3];
        float Short_TSL = params[tid * 5 + 4];

        
        int result_idx = tid * 16;
        float long_total_profit = 0.0;
        float long_total_loss = 0.0;
        float short_total_profit = 0.0;
        float short_total_loss = 0.0;
        int num_long_trades = 0;
        int num_short_trades = 0;
        float position = 0.0;
        float entry_price = 0.0;
        float highest_price = 0.0;
        float lowest_price = 0.0;

        // Loop through all candles (starting at 2 for safe previous value comparisons)
        for (int j = 2; j < num_candles; j++) {
            float close = close_price[j];
            float atr = atr_val[j];
            
            // Entry conditions using MACD-based signals:
            bool long_condition = (macd_line[j] > signal_line[j]) && (histogram[j] > long_HISTOGRAM) && (atr > ATR_value);
            bool short_condition = (macd_line[j] < signal_line[j]) && (histogram[j] < short_HISTOGRAM) && (atr > ATR_value);

            if (position == 0.0) {
                if (long_condition) {
                    position = 1.0;   // Open long trade.
                    entry_price = close;
                    highest_price = close; 
                    num_long_trades++;
                } else if (short_condition) {
                    position = 2.0;   // Open short trade.
                    entry_price = close;
                    lowest_price = close;
                    num_short_trades++;
                }
            } else if (position == 1.0) {
                highest_price = fmaxf(highest_price, close);  
                bool long_exit_condition = (macd_line[j] < signal_line[j]) && (ema_5[j] < sma_34[j]);
                bool trailing_stop_loss = (close < (highest_price - Long_TSL));
                if (long_exit_condition || trailing_stop_loss) {
                    float profit = close - entry_price;
                    if (profit > 0.0)
                        long_total_profit += profit;
                    else
                        long_total_loss += profit;
                    position = 0.0;
                }
            } else if (position == 2.0) {
                lowest_price = fminf(lowest_price, close);  
                bool short_exit_condition = (macd_line[j] > signal_line[j]) && (ema_5[j] > sma_34[j]);
                bool trailing_stop_loss = (close > (lowest_price + Short_TSL));
                if (short_exit_condition || trailing_stop_loss) {
                    float profit = entry_price - close;
                    if (profit > 0.0)
                        short_total_profit += profit;
                    else
                        short_total_loss += profit;
                    position = 0.0;
                }
            }
            
            // Force close any open trade on the last candle.
            if (j == num_candles - 1 && position != 0.0) {
                float profit = (position == 1.0) ? (close - entry_price) : (entry_price - close);
                if (position == 1.0) {
                    if (profit > 0.0)
                        long_total_profit += profit;
                    else
                        long_total_loss += profit;
                } else if (position == 2.0) {
                    if (profit > 0.0)
                        short_total_profit += profit;
                    else
                        short_total_loss += profit;
                }
                position = 0.0;
            }
        }
        
        // Write the results out for this parameter set:
        results[result_idx] = long_HISTOGRAM;
        results[result_idx + 1] = Long_TSL;
        results[result_idx + 2] = num_long_trades;
        results[result_idx + 3] = long_total_profit;
        results[result_idx + 4] = long_total_loss;
        results[result_idx + 5] = (long_total_profit > fabsf(long_total_loss)) ? (fabsf(long_total_loss) / long_total_profit) * 100.0f : 0.0f;

        results[result_idx + 6] = short_HISTOGRAM;
        results[result_idx + 7] = Short_TSL;
        results[result_idx + 8] = num_short_trades;
        results[result_idx + 9] = short_total_profit;
        results[result_idx + 10] = short_total_loss;
        results[result_idx + 11] = (short_total_profit > fabsf(short_total_loss)) ? (fabsf(short_total_loss) / short_total_profit) * 100.0f : 0.0f;

        results[result_idx + 12] = num_long_trades + num_short_trades;
        float total_profit = long_total_profit + short_total_profit;
        float total_loss = fabsf(long_total_loss + short_total_loss);
        results[result_idx + 13] = (num_long_trades + num_short_trades) > 0 ? (total_profit / (total_profit + total_loss)) * 100.0f : 0.0f;
        results[result_idx + 14] = total_profit - total_loss;
        results[result_idx + 15] = ATR_value;
    }
    ''',
    'BACKTEST_KERNEL'
)



def BACKTEST_GPU(df, params):
    num_candles = len(df)
    num_params = len(params)
    results = cp.zeros((num_params * 16), dtype=cp.float32)
    threads_per_block = 256
    blocks_per_grid = (num_params + threads_per_block - 1) // threads_per_block

    gpu_data = {
        'atr': cp.asarray(df['atr'], dtype=cp.float32),
        'close': cp.asarray(df['close'], dtype=cp.float32),
        'ema_5': cp.asarray(df['ema_5'], dtype=cp.float32),
        'sma_34': cp.asarray(df['sma_34'], dtype=cp.float32),
        'macd_line': cp.asarray(df['macd_line'], dtype=cp.float32),
        'signal_line': cp.asarray(df['signal_line'], dtype=cp.float32),
        'histogram': cp.asarray(df['histogram'], dtype=cp.float32),
        'params': cp.asarray(params, dtype=cp.float32)
    }
    
    BACKTEST_KERNEL((blocks_per_grid,), (threads_per_block,), (
        gpu_data['atr'],
        gpu_data['close'],
        gpu_data['ema_5'],
        gpu_data['sma_34'],
        gpu_data['macd_line'],
        gpu_data['signal_line'],
        gpu_data['histogram'],
        gpu_data['params'],
        results,
        num_candles,
        num_params
    ))
    return results.get().reshape((num_params, 16))

def TRADING_BOT(df):
    df['atr'] = ATR_CALCULATOR(df, 14)
    df['ema_5'] = CALCULATE_EMA(df, 5)
    df['sma_34'] = CALCULATE_SMA(df, 34)
    df = CALCULATE_MACD(df,5,34,9)
    
    params = np.array([
        (long_HISTOGRAM, short_HISTOGRAM, ATR_value, Long_TSL, Short_TSL)
        for long_HISTOGRAM in range(0, 650, 50)
        for short_HISTOGRAM in range(0, 650, 50)
        for ATR_value in range(0, 600, 100)
        for Long_TSL in range(0, 800, 100)
        for Short_TSL in range(0, 800, 100)
    ], dtype=np.float32)
    
    start_time = datetime.now()
    results = BACKTEST_GPU(df, params)
    end_time = datetime.now()
    
    print(f'Backtesting completed in {end_time - start_time} seconds')
    df_results = pd.DataFrame(results, columns=[
        'Long Histogram', 
        'Long TSL',
        'Long Trades', 
        'Long Total Profit', 
        'Long Total Loss', 
        'Long Loss % from Profit',
        'Short Histogram',
        'Short TSL',
        'Short Trades',
        'Short Total Profit',
        'Short Total Loss',
        'Short Loss % from Profit',
        'Total Trades',
        'Win %',
        'Net Profit',
        'ATR'
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
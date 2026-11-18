import pandas as pd
import numpy as np
import pandas_ta as ta
import cupy as cp
from joblib import Memory
from datetime import datetime

# Set up caching for Python indicator calculations
memory = Memory("./cache", verbose=0)

# Heikin-Ashi calculation, returning both original OHLC and HA columns
def CALCULATE_HEIKINASHI(df):
    # Prepare HA DataFrame
    ha = pd.DataFrame(index=df.index, columns=['ha_open','ha_high','ha_low','ha_close'], dtype=float)
    # Compute HA close for all bars
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    # First bar: HA open, high, low
    ha.iloc[0, ha.columns.get_loc('ha_open')] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    ha.iloc[0, ha.columns.get_loc('ha_high')] = max(
        df['high'].iloc[0],
        ha.iloc[0, ha.columns.get_loc('ha_open')],
        ha.iloc[0, ha.columns.get_loc('ha_close')]
    )
    ha.iloc[0, ha.columns.get_loc('ha_low')] = min(
        df['low'].iloc[0],
        ha.iloc[0, ha.columns.get_loc('ha_open')],
        ha.iloc[0, ha.columns.get_loc('ha_close')]
    )
    # Remaining bars
    for i in range(1, len(df)):
        prev_o = ha.iloc[i-1, ha.columns.get_loc('ha_open')]
        prev_c = ha.iloc[i-1, ha.columns.get_loc('ha_close')]
        o = (prev_o + prev_c) / 2
        c = ha.iloc[i, ha.columns.get_loc('ha_close')]
        h = max(df['high'].iloc[i], o, c)
        l = min(df['low'].iloc[i], o, c)
        ha.iloc[i, ha.columns.get_loc('ha_open')] = o
        ha.iloc[i, ha.columns.get_loc('ha_high')] = h
        ha.iloc[i, ha.columns.get_loc('ha_low')] = l
    # Combine original OHLC with HA columns
    ha_full = pd.concat([df, ha], axis=1)
    return ha_full


@memory.cache
def CALCULATE_SMA(df, length):
    df['oc2'] = (df['ha_open'] + df['ha_close']) / 2
    return ta.sma(df['oc2'], length=length)

# CUDA kernel: MAX_VOL, MIN_VOL with Net Profit calculation inside
BACKTEST_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void BACKTEST(
    const float* high, const float* low,
    const float* ha_open, const float* ha_close,
    const float* sma7, const float* sma14,
    const float* oc2, const float* params,
    float* results,
    const int n, const int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) return;
    int base = tid * 17;
    float MAX_VOL = params[tid*4];
    float MIN_VOL = params[tid*4+1];
    float SHORT_CANDLE_HEIGHT = params[tid*4+2];
    float LONG_CANDLE_HEIGHT = params[tid*4+3];

    float long_profit=0, long_loss=0, short_profit=0, short_loss=0;
    int long_count=0, short_count=0;
    int total_count=0;
    float entry=0;
    int pos=0;

    for(int i=2;i<n;i++){
        float h0 = high[i], l0 = low[i];
        float s0 = sma7[i], sf0 = sma14[i];
        float c0 = ha_close[i], c1 = ha_close[i-1], c2 = ha_close[i-2];
        float o0 = ha_open[i], o1 = ha_open[i-1], o2 = ha_open[i-2];
        float max10 = s0, min10 = s0;
        for(int k=i-9;k<=i;k++) if(k>=0){ if(sma7[k]>max10) max10=sma7[k]; if(sma7[k]<min10) min10=sma7[k]; }
        bool v1 = (max10>s0)?((max10-s0)>MAX_VOL):((s0-max10)>MAX_VOL);
        bool v2 = (min10>s0)?((min10-s0)>MIN_VOL):((s0-min10)>MIN_VOL);
        bool vol_ok = v1 || v2;
        bool short_entry = (sf0>s0) && 
                                        (o2 - c0) > SHORT_CANDLE_HEIGHT && 
                                        (c0<c1 && c1<c2) && 
                                        (c2<o2) && 
                                        (c1<o1) && 
                                        (c0<o0 && o0<s0) && 
                                        vol_ok;
        bool long_entry  = (sf0<s0) && 
                                        (c0 - o2) > LONG_CANDLE_HEIGHT && 
                                        (c0>c1 && c1>c2) && 
                                        (c2>o2) && 
                                        (c1>o1) && 
                                        (c0>o0 && o0>s0) && 
                                        vol_ok;
        if(pos == 0){
            if(long_entry){ pos=1; entry=oc2[i]; long_count++; total_count++; }
            else if(short_entry){ pos=-1; entry=oc2[i]; short_count++; total_count++; }

        } else {
            float prof = (pos==1)? (l0 - entry) : (entry - h0);
            bool ex;
            
            ex = (pos==1)
                    ? (s0 < sf0)  // long exit when sma7 crosses below sma14
                    : (s0 > sf0); // short exit when sma7 crosses above sma14
            
            if(ex){
                if(pos==1){ if(prof>0) long_profit+=prof; else long_loss+=prof; }
                else      { if(prof>0) short_profit+=prof; else short_loss+=prof; }
                pos = 0;
            }
        }
        if(i==n-1 && pos!=0){
            float prof=(pos==1)?(c0-entry):(entry-c0);
            if(pos==1){ if(prof>0) long_profit+=prof; else long_loss+=prof; }
            else       { if(prof>0) short_profit+=prof; else short_loss+=prof; }
            pos=0;
        }
    }
    float tot_p = long_profit + short_profit;
    float tot_l = fabs(long_loss) + fabs(short_loss);
    float win_pct = total_count>0 ? (tot_p/(tot_p+tot_l))*100 : 0;
    float long_loss_pct = long_profit>fabs(long_loss) ? (fabs(long_loss)/long_profit)*100 : 0;
    float short_loss_pct= short_profit>fabs(short_loss)? (fabs(short_loss)/short_profit)*100 : 0;

    results[base+0]  = MAX_VOL;
    results[base+1]  = MIN_VOL;
    results[base+2]  = SHORT_CANDLE_HEIGHT;
    results[base+3]  = LONG_CANDLE_HEIGHT;
    results[base+4]  = long_profit;
    results[base+5]  = long_loss;
    results[base+6]  = long_loss_pct;
    results[base+7] = short_profit;
    results[base+8] = short_loss;
    results[base+9] = short_loss_pct;
    results[base+10]  = total_count;
    results[base+11]  = long_count;
    results[base+12]  = short_count;
    results[base+13]  = win_pct;
    results[base+14] = tot_p;
    results[base+15] = tot_l;
    results[base+16] = tot_p - tot_l;  
}
''', 'BACKTEST')

# Wrapper
def BACKTEST_GPU(df, params):
    n, m = len(df), len(params)
    arrs = [
        cp.asarray(df['high'], dtype=cp.float32),
        cp.asarray(df['low'],dtype=cp.float32),
        cp.asarray(df['ha_open'], dtype=cp.float32),
        cp.asarray(df['ha_close'],dtype=cp.float32),
        cp.asarray(df['sma_7'], dtype=cp.float32),
        cp.asarray(df['sma_14'], dtype=cp.float32),
        cp.asarray(df['oc2'],dtype=cp.float32),
        cp.asarray(params.flatten(),dtype=cp.float32)
    ]
    out = cp.zeros(m*17, dtype=cp.float32)
    threads, blocks = 256, (m+255)//256
    BACKTEST_KERNEL((blocks,), (threads,), (*arrs, out, n, m))
    return out.get().reshape(m,17)

# Main
if __name__=='__main__':
    INPUT_FILE = r"15min ohlc MARCH.csv"
    OUTPUT_FILE_NET_PROFIT = r"15min_backtest_v22.csv"
    
    df = pd.read_csv(INPUT_FILE)
    ha = CALCULATE_HEIKINASHI(df)
    ha['oc2']   = (ha['open'] + ha['close'])/2
    ha['sma_7'] = CALCULATE_SMA(ha,7)
    ha['sma_14'] = CALCULATE_SMA(ha,14)

    params = np.array([
        (maxv, minv, lch, sch)
        for maxv in range(0,500,100)
        for minv in range(0,500,100)
        for sch in range(0,1000,100)
        for lch in range(0,1000,100)
    ], dtype=np.float32)

    start = datetime.now()
    res = BACKTEST_GPU(ha, params)
    print("Duration:", datetime.now() - start)

    cols = [
        'MAX_VOL', 'MIN_VOL', 'SHORT_CANDLE_HEIGHT','LONG_CANDLE_HEIGHT',
        'Long Profit','Long Loss','Long Loss %',
        'Short Profit','Short Loss','Short Loss %',
        'Total Trades', 'Long Trades', 'Short Trades', 'Win %',
        'Total Profit','Total Loss','Net Profit'
    ]
    df_result = pd.DataFrame(res, columns=cols)
    df_result = df_result.sort_values('Net Profit', ascending=False)
    df_result.to_csv(OUTPUT_FILE_NET_PROFIT, index=False)
    print(df_result)
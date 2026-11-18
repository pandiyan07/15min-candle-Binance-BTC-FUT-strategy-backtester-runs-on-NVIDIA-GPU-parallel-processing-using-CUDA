# GPU-Accelerated Crypto/Stock Trading Strategy Backtester


## Table of Contents
1. [About The Project](#about-the-project)
2. [Built With](#built-with)
3. [Key Features](#key-features)
4. [Getting Started](#getting-started)
   - [Technologies Used (Prerequisites)](#technologies-used-prerequisites)
   - [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [How It Works](#how-it-works)
8. [Main Python Functions](#main-python-functions)
9. [Output that can be expected](#output-that-can-be-expected)
10. [Configuration](#configuration)
11. [Version History (changes between each version)](#version-history-changes-between-each-version)
12. [Future Roadmap](#future-roadmap)
13. [License Description](#license-description)
14. [Disclaimer](#disclaimer)

---

## About The Project
This repository contains a high-performance quantitative trading backtesting suite designed to run parameter optimization sweeps across massive historical 15 minute OHLC datasets. By leveraging NVIDIA CUDA via **CuPy**, the framework accelerates multi-parameter evaluations for trading algorithms—ranging from Moving Average (EMA/SMA) crossovers and MACD variations to advanced Heikin-Ashi pattern recognition and custom multi-candle breakout rules.

---

## Built With
* Python
* CuPy (CUDA GPU Acceleration)
* Pandas & Pandas-TA
* NumPy
* Joblib

---

## Key Features
* **GPU Acceleration:** Offloads computationally intensive iterative backtesting loops to the GPU using custom CUDA kernels (`cp.RawKernel`), achieving massive speedups over traditional CPU-based backtesters.
* **Extensive Parameter Sweeps:** Automatically iterates through hundreds or thousands of parameter combinations (such as ATR thresholds, volatility bounds, MA deviations, and trailing stop-losses).
* **Multi-Strategy Support:** Includes implementations for standard candlestick indicators, Heikin-Ashi conversions, and multi-bar momentum logic.
* **Multi-Month Aggregation:** Tools like `net profit totaller.py` allow seamless merging and summing of backtest performance across multi-month CSV exports.

---

## Getting Started

### Technologies Used (Prerequisites)
* Python 3.x
* NVIDIA GPU with CUDA Toolkit installed and compatible drivers
* Python Libraries: `cupy`, `pandas`, `pandas_ta`, `numpy`, `joblib`

### Installation
1. Clone the repository:
```bash
git clone [https://github.com/pandiyan07/15min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git](https://github.com/pandiyan07/15min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA.git)
cd 15min-candle-Binance-BTC-FUT-strategy-backtester-runs-on-NVIDIA-GPU-parallel-processing-using-CUDA
```

2. Install the required Python dependencies:
```bash
pip install pandas numpy pandas-ta cupy-cuda12x joblib

```
*(Note: Ensure your CuPy installation matches your installed CUDA Toolkit version).*

---

## Usage

1. Place your historical OHLC CSV file (e.g., `5min ohlc MARCH.csv` or `15min ohlc MARCH.csv`) in the working directory.
2. Run your chosen version script directly via Python:
```bash
python "cuda 15min v27 - entry cond changed in v26.py"

```
3. The script will execute the GPU kernel, print execution duration and top-performing parameter combinations, and output the sorted results into a target CSV file.

---

## Project Structure

```text
├── cache/                                      # Joblib caching directory for indicator calculations
├── 5min ohlc MARCH.csv                         # Sample input market data
├── net profit totaller.py                      # Multi-month result aggregation utility
├── Heikanashi trade entry and exit rules.txt   # Strategy logic documentation text file
└── cuda 15min v1.py to v30.py                  # Individual strategy and version-controlled backtesting scripts

```

---

## How It Works

1. **Data Preprocessing & Indicators:** Historical data is loaded using Pandas. Technical indicators (`EMA`, `SMA`, `MACD`, `ATR`, or custom Heikin-Ashi calculations) are computed and cached.
2. **Parameter Grid Generation:** NumPy arrays are populated with multi-dimensional parameter tuples (e.g., variation limits, volatility boundaries, and trailing stop values).
3. **GPU Kernel Execution:** Data arrays and parameter sets are transferred to the GPU memory. A parallel CUDA kernel assigns individual threads to evaluate unique parameter combinations across the entire timeline of candles.
4. **Performance Aggregation:** Results (trades, profits, losses, win rates, and net profit) are written back to host memory, converted into a Pandas DataFrame, sorted, and exported to CSV.

---

## Main Python Functions

* `CALCULATE_HEIKINASHI(df)`: Computes smoothed Heikin-Ashi open, high, low, and close values.


* `CALCULATE_EMA(df, length)` / `CALCULATE_SMA(df, length)`: Computes Exponential and Simple Moving Averages based on typical price (`oc2`).


* `CALCULATE_MACD(df, fa, sl, sig)`: Generates MACD line, signal line, and histogram metrics.


* `ATR_CALCULATOR(df, length)`: Calculates Average True Range for volatility filtering.


* `BACKTEST_GPU(df, params)`: Manages array transfers to GPU memory, grid/block configuration, and kernel launching.



---

## Output that can be expected

Execution prints performance diagnostics to the console, including execution time and a sorted Pandas DataFrame table displaying columns such as:

* Parameter configurations (e.g., `MAX_VOL`, `MIN_VOL`, `ATR`, `TSL`)
* `Total Trades`, `Long Trades`, `Short Trades`
* `Win %` / `Win Rate (%)`
* `Long Profit`, `Long Loss`, `Short Profit`, `Short Loss`
* `Net Profit` (sorted in descending order)

A corresponding CSV file containing the complete sweep results is saved locally.

---

## Configuration

Inputs, input file paths, and output destinations can be configured directly inside the `if __name__ == "__main__":` block of each script:

* `INPUT_FILE`: Path to the historical OHLC CSV dataset.
* `OUTPUT_FILE_NET_PROFIT`: Target path for saving sorted backtest results.
* Parameter ranges (`range(...)` loops in `TRADING_BOT`) can be adjusted to narrow or expand optimization grids.

---

## Version History (changes between each version)

| Python File / Script | Description / Version Changes |
| --- | --- |
| [`cuda 5min v10.py`](https://www.google.com/search?q=cuda%25205min%2520v10.py) | Refined GPU backtester utilizing a 5 EMA and 13 SMA crossover system, incorporating consecutive trend-bar checks and strict entry variation boundaries. |
| [`cuda 15min v1.py`](https://www.google.com/search?q=cuda%252015min%2520v1.py) | Initial EMA 5 / SMA 34 crossover framework with entry variation rules and trailing stop-loss (TSL). |
| [`cuda 15min v2.py`](https://www.google.com/search?q=cuda%252015min%2520v2.py) | Tests separate SMA 34 entry and exit variation parameters.|
| [`cuda 15min v3.py`](https://www.google.com/search?q=cuda%252015min%2520v3.py) | Base crossover script optimized around static TSL execution paths.|
| [`cuda 15min v4.py`](https://www.google.com/search?q=cuda%252015min%2520v4.py) | Adds MA deviation and SMA 34 variation conditions with independent long/short TSL controls.|
| [`cuda 15min v5.py`](https://www.google.com/search?q=cuda%252015min%2520v5.py) | Restructured parameter ordering and output formatting. |
| [`cuda 15min v6 deviation.py`](https://www.google.com/search?q=cuda%252015min%2520v6.py) | Focuses specifically on long/short MA deviation parameters paired with trailing stops.|
| [`cuda 15min v7 variation.py`](https://www.google.com/search?q=cuda%252015min%2520v7.py) | Incorporates SMA 34 variation checks alongside trailing stop-losses.|
| [`cuda 15min v8.py`](https://www.google.com/search?q=cuda%252015min%2520v8.py) | Permits simultaneous long and short active trades with independent trailing stop management.|
| [`cuda 15min v9 with atr.py`](https://www.google.com/search?q=cuda%252015min%2520v9.py) | Integrates Average True Range (ATR) threshold filtering into multi-condition entries.|
| [`cuda 15min v10.py`](https://www.google.com/search?q=cuda%252015min%2520v10.py) | Refines parallel long/short trade execution logic with integrated ATR and MA deviation filters.|
| [`cuda 15min v11 MACD with static TSL.py`](https://www.google.com/search?q=cuda%252015min%2520v11%2520MACD%2520with%2520static%2520TSI.py) | Combines MACD line/signal crossovers and histogram thresholds with static TSL exits.|
| [`cuda 15min v12 MACD only.py`](https://www.google.com/search?q=cuda%252015min%2520v12%2520MACD%2520only.py) | Streamlines execution to single-position active trades governed purely by MACD signal crosses.|
| [`cuda 15min v13 - trading v12 simultaneously.py`](https://www.google.com/search?q=cuda%252015min%2520v13%2520-%2520trading%2520v12%2520simultaneously.py) | Extends MACD strategy to permit independent, simultaneous long and short positions.|
| [`cuda 15min v14 - new v5 with 21sma.py`](https://www.google.com/search?q=cuda%252015min%2520v14%2520-%2520new%2520v5%2520with%252021sma.py) | Incorporates a 21 SMA benchmark alongside EMA 5 for faster deviation and trend evaluation.|
| [`cuda 15min v15 - v12 with 21sma.py`](https://www.google.com/search?q=cuda%252015min%2520v15%2520-%2520v12%2520with%252021sma.py) | Integrates MACD entries with a 21 SMA baseline for exit confirmation signals.|
| [`cuda 15min v16 - v12 with 13sma & sumof_last_four_histogram.py`](https://www.google.com/search?q=cuda%252015min%2520v16%2520-%2520v12%2520with%252013sma%2520%2526%2520sumof_last_four_histogram.py) | Uses a 13 SMA line and incorporates a cumulative sum of the last four MACD histogram bars.|
| [`cuda 15min v17 - variation using 13sma.py`](https://www.google.com/search?q=cuda%252015min%2520v17%2520-%2520variation%2520using%252013sma.py) | Evaluates entry variations based on 13 SMA slope differentials combined with ATR filters.|
| [`cuda 15min v18 - v15 with 13macd.py`](https://www.google.com/search?q=cuda%252015min%2520v18%2520-%2520v15%2520with%252013macd.py) | Adjusts MACD fast/slow periods to a (5, 13, 9) configuration for faster responsiveness.|
| [`cuda 15min v19 - v17 with tsl.py`](https://www.google.com/search?q=cuda%252015min%2520v19%2520-%2520v17%2520with%2520tsl.py) | Appends custom trailing stop-loss metrics to the 13 SMA variation framework.|
| [`cuda 15min v20 - v17 with 3 candle tecnique.py`](https://www.google.com/search?q=cuda%252015min%2520v20%2520-%2520v17%2520with%25203%2520candle%2520technique.py) | Implements a specialized multi-candle open-close weighting rule across consecutive bars.|
| [`cuda 15min v21 - heiknashi with 3 candle technique.py`](https://www.google.com/search?q=cuda%252015min%2520v21%2520-%2520heiknashi%2520with%25203%2520candle%2520technique.py) | Converts data to Heikin-Ashi candles, applying 7 SMA slope rules and 3-candle confirmation patterns.|
| [`cuda 15min v22 - heiknashi 7sma.py`](https://www.google.com/search?q=cuda%252015min%2520v22%2520-%2520heiknashi%25207sma.py) | Combines Heikin-Ashi 7 SMA conditions, ATR thresholds, and dynamic profit-based exit rules.|
| [`cuda 15min v23 - heiknashi 7-14 ma-crossover.py`](https://www.google.com/search?q=cuda%252015min%2520v23%2520-%2520heiknashi%25207-14%2520ma-crossover.py) | Uses a 7 and 14 SMA crossover system on Heikin-Ashi data with custom volatility parameter bounds.|
| [`cuda 15min v24 - heiknashi v23 entry condition v22 exit condition.py`](https://www.google.com/search?q=cuda%252015min%2520v24%2520-%2520heiknashi%2520v23%2520entry%2520condition%2520v22%2520exit%2520condition.py) | Merges v23 entry rules with v22 profit-percentage target exit mechanisms.|
| [`cuda 15min v25 - heiknashi and normal candles combined.py`](https://www.google.com/search?q=cuda%252015min%2520v25%2520-%2520heiknashi%2520and%2520normal%2520candles%2520combined.py) | Evaluates conditions blending raw OHLC candle heights with Heikin-Ashi trend metrics.|
| [`cuda 15min v26 - crossover exit only in v25.py`](cuda%2015min%20v26%20-%20crossover exit only in v25.py) | Streamlines v25 to use strict MA crossover exit triggers.|
| [`cuda 15min v27 - entry cond changed in v26.py`](cuda%2015min%20v27%20-%20entry cond changed in v26.py) | Refines entry calculations using specific open/close differences from prior candles.|
| [`cuda 15min v28 - modified v27.py`](cuda%2015min%20v28%20-%20modified v27.py) | Incorporates dynamic volatility threshold checks, strict high/low candle boundaries, and special large-candle triggers.|
| `cuda 15min v29 - volatality removed from v27.py` | Strips out standard volatility filters to evaluate pure candle height parameters.|
| `cuda 15min v30 - simultaneous trade in v27.py`| Implements concurrent independent long and short trade tracking on top of v27 logic.|
| [`net profit totaller.py`](https://www.google.com/search?q=net%2520profit%2520totaller.py) | Aggregates and sums net profit results across multi-month CSV exports.|
| [`Heikanashi trade entry and exit rules.txt`](Heikanashi%20trade entry and exit rules.txt) | Reference document outlining rule definitions for Heikin-Ashi executions.|

---

## Future Roadmap

* Add support for multi-GPU scaling to handle larger parameter grids concurrently.
* Integrate live broker API connectors (e.g., Interactive Brokers, Binance) for automated live deployment of top-performing parameters.
* Implement automated walk-forward optimization and out-of-sample cross-validation.

---

## License Description

This project is open-source software provided under standard repository usage guidelines. Feel free to modify and adapt the scripts for personal or research purposes.

---

## Disclaimer

Trading financial instruments, stocks, and cryptocurrencies carries a high level of risk and may not be suitable for all investors. Backtesting results generated by these scripts are hypothetical and do not guarantee future live trading performance. The author assumes no responsibility for any financial losses incurred from using these scripts. Always test strategies thoroughly in a simulated environment before deploying real capital.

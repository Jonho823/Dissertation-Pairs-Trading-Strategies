# Dissertation-Pairs-Trading-Strategies
September 2023

Student: Ka Fai Jonathan Ho

## Introduction
This is my dissertation source code for pairs trading strategies, exploring the distance, cointegration, and copula methods on FTSE 100 data from Jan 2013 to Jun 2023. The execution involves identifying co-moving pairs, initiating positions on spread divergence, and closing positions based on predetermined exit criteria.

## Data
The dissertation utilizes a dataset comprising daily stock data from all constituents of the FTSE100 index, sourced from Yahoo Finance from January 2013 to June 2023. Data extraction was conducted using the yFinance Python module.

## Installation Instruction
1. Clone this project
2. Import **`ftse100_data_extraction.py`** module to download FTSE stock data into a CSV file.
3. Utilize the **`distance_approach.py`**, **`cointegration_approach.py`**, and **`copula_approach.py`** modules to access functions for executing backtesting on pairs trading strategies based on different approaches.
4. Evaluate the profitability of pairs trading strategies by employing the **`performance_measurement.py`** module to calculate performance metrics and generate reports.
5. Refer to **`main.py`** for a demonstration of the execution process.
6. Explore the documentation of various pairs trading strategies stored in **`pairs_trading_doc.ipynb`** for detailed insights and explanations.

## Coding Contributions
[1] H. Darbinyan (2021) Distance Method Complete Algo [Source code]. https://github.com/haykdb/thesis_2021/blob/main/Disrance%20Method%20Complete%20Algo.ipynb.

[2] Financialnoob (2022) Pairs trading. Pairs selection. Distance [Source code]. https://github.com/financialnoob/pairs_trading/blob/main/2.pairs_trading.pairs_selection.distance.ipynb

[3] H. Darbinyan (2021) Cointegration [Source code]. https://github.com/haykdb/thesis_2021/blob/main/Cointegration.ipynb

[4] Financialnoob (2022) Pairs trading. Pairs selection. Cointegration [Source code]. https://github.com/financialnoob/pairs_trading/blob/main/4.pairs_trading.pairs_selection.cointegration_1.ipynb

[5] H.R. Xu (2023) Pairs Trading Copula [Source code]. https://github.com/XUNIK8/Pairs-Trading-Copula/blob/main/Final-Copula.ipynb.

## Disclaimer
This repository contains code developed for the backtesting of my dissertation project. However, due to ownership rights held by my university, I am unable to distribute the paper associated with this work. Therefore, this repository solely serves as a demonstration of the code's functionality and is not accompanied by the full academic context.

Please note:
- This code is provided as-is, without any warranty or guarantee of its suitability for any purpose.
- Users should exercise caution and ensure compliance with any relevant academic or institutional policies regarding the sharing and use of this code.
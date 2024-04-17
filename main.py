#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This documentation provides an overview and demonstration of the module incorporation process. T
he purpose is to showcase the functionality and usage of the incorporated modules.

Due to cost of computational, the demonstration employs rolling windows covering 13 periods.

Parameter setting:
    n = 5, select top 5 pairs for the periods
    start formation date = 2021-01-01
    end formation date = 2022-12-30
    start trading date = 2022-01-01
    end trading date = 2023-06-30

Note:
The data extraction step is currently commented out to avoid unnecessary execution, as it does not need to be repeated every time.
"""
#%%
#%load_ext autoreload
#%autoreload 2

# import ftse100_data_extraction as ftsede # module to download data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import distance_approach as dist
import cointegration_approach as coint
import copula_approach as copula
import performance_measurement as pm
import time

#%% Set up parameters and download the ftse100 stock data

# =============================================================================
# ftse_symbols = ftsede.get_ftse100_symbols()
# start_date = '2021-01-01'
# end_date = '2023-06-30'
# output_file = 'ftse100_2021_2023_06.csv'
# 
# ftsede.download_stock_data(ftse_symbols, start_date, end_date, output_file)
# =============================================================================

#%% Load dataframe as df

df = pd.read_csv('ftse100_2021_2023_06.csv', index_col = 'Date')

#%% Load the date for formation period and trading period

date_path = 'date_data.csv' # monthly rotation
dates = pd.read_csv(date_path)

#print(type(dates['date1']))

# In[13]: Run the Distance approach with period in index

# time - 1005 secs for 13 periods (top 5), 1 min for 1 period (top 5)

#start time counter
start_time = time.time()

distance_returns = pd.DataFrame(index = dates.index, columns = ['returns','date1','date2','date3','date4'])
for i in dates.index[96:]:
    date1 = dates.loc[i, 'date1']
    date2 = dates.loc[i, 'date2']
    date3 = dates.loc[i, 'date3']
    date4 = dates.loc[i, 'date4']
    dis_seq = dist.distance_method(date1, date2, date3, date4, 5, df, 10000) # top 5 min. ssd
    value = dist.average(dis_seq)
    distance_returns.loc[i, 'returns'] = value
    distance_returns.loc[i, 'date1'] = dates.loc[i, 'date1']
    distance_returns.loc[i, 'date2'] = dates.loc[i, 'date2']
    distance_returns.loc[i, 'date3'] = dates.loc[i, 'date3']
    distance_returns.loc[i, 'date4'] = dates.loc[i, 'date4']

#do end time - start time
output = "%2f" % (time.time() - start_time)
#print output
print(output + " seconds")

# error: unsupported str, coz changed not listed firms price into -, replace with 0
# error: test return result with 0, maybe ssd found those with 0 close price, need to eliminate those
# fix error: drop all columns that has 0 in formation function above
# try 104:105, code normal

#%% Extract top 5 distance approach result as excel

distance_returns.to_excel('top_5_pairs_returns_DM4.xlsx', index = True)

#%% Plot the Distance approach and save it

dist.plot_pairs_distance(df, date1, date2, date3, date4, N=1)
plt.savefig('top_1_pairs_DM4.png')

#%% Display calculation metrices for distance approach

distance_formation, distance_trading = dist.calculate_metrics_distance(df, date1, date2, date3, date4)

#%% measure the comprehensive performance for copula approach

distance_performance_df = pm.performance_report('top_5_pairs_returns_DM4.xlsx')

#%% Run the Cointegration approach with period in index

# time - 30 mins for 13 periods (top5), for 2 mins for 1 period (top5)

#start time counter
start_time = time.time()

coint_returns = pd.DataFrame(index = dates.index, columns = ['returns','date1','date2','date3','date4'])
for i in dates.index[96:]:
    date1 = dates.loc[i, 'date1']
    date2 = dates.loc[i, 'date2']
    date3 = dates.loc[i, 'date3']
    date4 = dates.loc[i, 'date4']
    coint_seq = coint.Coint_Trading(date1, date2, date3, date4, 5, df, 10000) # top 5 pair
    value = coint.average(coint_seq)
    coint_returns.loc[i, 'returns'] = value
    coint_returns.loc[i, 'date1'] = dates.loc[i, 'date1']
    coint_returns.loc[i, 'date2'] = dates.loc[i, 'date2']
    coint_returns.loc[i, 'date3'] = dates.loc[i, 'date3']
    coint_returns.loc[i, 'date4'] = dates.loc[i, 'date4']
    
#do end time - start time
output = "%2f" % (time.time() - start_time)
#print output
print(output + " seconds") # time 1 min per mth (top 20) 1.5 mins (top 5)

#%% Extract top 5 cointegration approach result as excel

coint_returns.to_excel('top_5_pairs_returns_coint.xlsx', index = True)

#%% Plot the Cointegration approach and save it

coint.plot_pairs_cointegration(df, date1, date2, date3, date4, N=1)
plt.savefig('top_1_pairs_coint.png')

#%% #%% Display calculation metrices for cointegration approach

#start time counter
start_time = time.time()

cointegration_formation, cointegration_trading = coint.calculate_metrics_cointegration(df, date1, date2, date3, date4)

#do end time - start time
output = "%2f" % (time.time() - start_time)
#print output
print(output + " seconds") # time 1.25 mins (top 1)

#%% measure the comprehensive performance for copula approach

cointegration_performance_df = pm.performance_report('top_5_pairs_returns_coint.xlsx')

#%% Run the Copula approach with period in index

# Read and preprocess data

filename = 'ftse100_2021_2023_06.csv'
stocks, df_prices = copula.read_stocks_prices(filename)
df_data = copula.calc_logReturns(df_prices)

#%%
#start time counter
# time - 75 secs for 13 periods (top5), 7 secs for 1 period (top5)

start_time = time.time()

copula_returns = pd.DataFrame(index = dates.index, columns = ['returns','date1','date2','date3','date4'])
for i in dates.index[96:]:
    date1 = dates.loc[i, 'date1']
    date2 = dates.loc[i, 'date2']
    date3 = dates.loc[i, 'date3']
    date4 = dates.loc[i, 'date4']
    cop_seq = copula.copula_approach(stocks, df_data, date1, date2, date3, date4, 5, 0.5) # seq = 5 pairs return
    df_portfolio_return = pd.concat(cop_seq, axis=1).mean(axis=1)
    df_portfolio_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_portfolio_return.dropna(inplace=True)
    value = pd.Series(np.cumprod(df_portfolio_return+1))[-1]-1
    copula_returns.loc[i, 'returns'] = value * 100
    copula_returns.loc[i, 'date1'] = dates.loc[i, 'date1']
    copula_returns.loc[i, 'date2'] = dates.loc[i, 'date2']
    copula_returns.loc[i, 'date3'] = dates.loc[i, 'date3']
    copula_returns.loc[i, 'date4'] = dates.loc[i, 'date4']

#do end time - start time
output = "%2f" % (time.time() - start_time)
#print output
print(output + " seconds") # time 7 secs (top 5)

#%% Extract top 5 copula approach result as excel

copula_returns.to_excel('top_5_pairs_returns_copula.xlsx', index = True)

#%%

copula.plot_pairs_copula(df_data, stocks, date1, date2, date3, date4, 0.5)

#%%

copula_trading_results = copula.calculate_metrics_copula(df_data, stocks, date1, date2, date3, date4, 0.5, n=2)

#%% measure the comprehensive performance for copula approach

copula_performance_df = pm.performance_report('top_5_pairs_returns_copula.xlsx')

#%%

def combine_dataframes(distance, coint, copula):
    # Transpose rows and columns
    distance = distance.rename(columns={'Strategy': 'Distance Approach'})
    distance_transposed = distance.transpose()
    coint = coint.rename(columns={'Strategy': 'Cointegration Approach'})
    coint_transposed = coint.transpose()
    copula = copula.rename(columns={'Strategy': 'Copula Approach'})
    copula_transposed = copula.transpose()
    
    # Combine transposed dataframes
    combined_df = pd.concat([distance_transposed, coint_transposed, copula_transposed], axis=0)
    
    return combined_df

#%%

combined_dataframe = combine_dataframes(distance_performance_df, cointegration_performance_df, copula_performance_df)
combined_dataframe.to_excel('profitability_3_approaches.xlsx', index = True)

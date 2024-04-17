#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module used to store the function to implement:
    - distance approach proposed by 
    Gatev et al., 2006 in "Pairs trading: Performance of a relative-value arbitrage rule."
    
"""

#%% Import libraries

import pandas as pd
import numpy as np
import statistics as stat
import warnings
warnings.filterwarnings('ignore')
# import time

#%% Define functions for operations

# Slice data into formation, trading period
# date format: year-month-day
def slicing_formation(date1, date2, df):
    # df = pd.read_csv(path, index_col = 'Date')
    dta = df.loc[date1:date2, (df != 0).all()]
    return(dta)

def slicing_trading(dta1, dta2, df):
    # df = pd.read_csv(path, index_col = 'Date')
    dta = df.loc[dta1 : dta2]
    return(dta)

# Normalize prices
def normalizing(data): # input: sliced dataset, output: normalized dataset
    df = pd.DataFrame(index = data.index, columns = data.columns)
    for i in df.columns:
        df[i][0] = 1
        for j in range(1, len(df)):
            df[i][j] = data[i][j] / data[i][0]
    return(df)

# Compute sum of squared difference
def ssd(col1, col2):
    s = (col1 - col2)**2
    ssd = s.sum()
    return(ssd)

def ssd_matrix(dta): # input normalized dataset
    matrix = pd.DataFrame(index = dta.columns, columns = dta.columns)
    for i in matrix.index:
        for j in matrix.columns:
            if i != j:
                matrix.loc[i,j] = ssd(dta[i], dta[j])
    return(matrix)

def min_values(matrix): # input: ssd matrix
    minimums = []
    for i in matrix.index:
        for j in matrix.columns:
            minimums.append(matrix.loc[i,j])
    minimums.remove(minimums[0])
    minimums = list(dict.fromkeys(minimums))
    minimums.sort()
    return(minimums)

def standard_deviation(col1,col2):
    series = col1 - col2
    std = stat.stdev(series)
    return(std)

# In[3]: Functions for pairs formation and return

def forming_pairs(date1, date2, df):
    dataframe = slicing_formation(date1, date2, df)
    norm_dataframe = normalizing(dataframe)
    coef_matrix = ssd_matrix(norm_dataframe)
    coef_list = min_values(coef_matrix)
    return(coef_matrix, coef_list)

def pair_tickers(coeff_matrix, coeff_list, x):
    pair_names = []
    for c in coeff_list:
        for i in coeff_matrix.index:
            for j in coeff_matrix.columns:
                if coeff_matrix.loc[i,j] == c:
                    names = [i,j]
                    pair_names.append(names)
    for i in pair_names:
        pair_names.remove([i[1], i[0]])
    return(pair_names[:x])

# =============================================================================
# def pair_tickers(coeff_matrix, coeff_list, x, df, date1, date2, date3, date4):
#     pair_names = []
#     for c in coeff_list:
#         for i in coeff_matrix.index:
#             for j in coeff_matrix.columns:
#                 if coeff_matrix.loc[i, j] == c:
#                     names = [i, j]
#                     pair_names.append(names)
#     for pair in pair_names:
#         if [pair[1], pair[0]] in pair_names:
#             pair_names.remove([pair[1], pair[0]])
#     return pair_names[:x]
# =============================================================================

def pair_profit(profit1, profit2):
    profit = (profit1 + profit2) / 2
    return(profit)

# In[4]: Trading Signals

def trading_df(pair1, pair2, date1, date2, date3, date4, df):
    # slice trading period and reset
    dataframe_trading = slicing_trading(date3, date4, df)
    dataframe_copy = dataframe_trading.copy()
    norm_dataframe_copy = normalizing(dataframe_copy)
    norm_dataframe_copy = norm_dataframe_copy.reset_index()
    # slice formation period
    dataframe_formation = slicing_formation(date1, date2, df)
    norm_dataframe_formation = normalizing(dataframe_formation)
    pair_frame = norm_dataframe_copy[['Date', pair1, pair2]]
    # compute standard deviation of pairs in formation period
    stdev = standard_deviation(norm_dataframe_formation[pair1], norm_dataframe_formation[pair2])
    pair_frame['Signal'] = pd.Series(np.zeros(len(pair_frame)))
    first_pair = pd.DataFrame(pair_frame['Date'])
    first_pair['Signal'] = pd.Series(np.zeros(len(first_pair)))
    second_pair = pd.DataFrame(pair_frame['Date'])
    second_pair['Signal'] = pd.Series(np.zeros(len(second_pair)))
    pair_frame['Signal'][0] = 0
    first_pair['Signal'][0] = 0
    second_pair['Signal'][0] = 0
    for x in range(len(pair_frame)):
        if pair_frame[pair1][x] - pair_frame[pair2][x] > 2 * stdev:
            pair_frame['Signal'][x] = 1
            first_pair['Signal'][x] = 1
            second_pair['Signal'][x] = -1
        elif pair_frame[pair2][x] - pair_frame[pair1][x] > 2 * stdev:
            pair_frame['Signal'][x] = 1
            first_pair['Signal'][x] = -1
            second_pair['Signal'][x] = 1
        elif pair_frame[pair2][x] - pair_frame[pair1][x] == 0:
            pair_frame['Signal'][x] = 0
            first_pair['Signal'][x] = 0
            second_pair['Signal'][x] = 0
        elif (pair_frame[pair1][x - 1] - pair_frame[pair2][x - 1]) / (pair_frame[pair1][x] - pair_frame[pair2][x]) < 0:
            pair_frame['Signal'][x] = 0
            first_pair['Signal'][x] = 0
            second_pair['Signal'][x] = 0
        else:
            pair_frame['Signal'][x] = pair_frame['Signal'][x - 1]
            first_pair['Signal'][x] = first_pair['Signal'][x - 1]
            second_pair['Signal'][x] = second_pair['Signal'][x - 1]
    pair_frame['Position'] = pair_frame['Signal'].diff()
    first_pair['Position'] = first_pair['Signal'].diff()
    second_pair['Position'] = second_pair['Signal'].diff()
    close1 = dataframe_trading[pair1]
    first_pair.index = first_pair['Date']
    first_pair['Close'] = close1
    first_pair.drop(labels = 'Date', axis = 1, inplace = True)
    first_pair = first_pair.reset_index()
    close2 = dataframe_trading[pair2]
    second_pair.index = second_pair['Date']
    second_pair['Close'] = dataframe_trading[pair2]
    second_pair.drop(labels = 'Date', axis = 1, inplace = True)
    second_pair = second_pair.reset_index()
    return(first_pair, second_pair)


# In[9]: Trading Strategy

# give dataframe with 'Signal', 'Position', 'Close', 'Date'
def trading(dta, capital):
    dta['Total Assets'] = pd.Series(np.zeros(len(dta)))
    dta['Holding'] = pd.Series(np.zeros(len(dta)))
    dta['Employed Capital'] = pd.Series(np.zeros(len(dta)))
    dta['Shorted Capital'] = pd.Series(np.zeros(len(dta)))
    dta['Free Capital'] = pd.Series(np.zeros(len(dta)))
    dta['Total Assets'][0] = capital
    dta['Holding'][0] = 0
    dta['Employed Capital'][0] = 0
    dta['Shorted Capital'][0] = 0
    dta['Free Capital'][0] = capital
    shorted_cap = 0
    for x in range(1, len(dta)):
        if dta['Position'][x] == 1:
            if dta['Holding'][x - 1] == 0:
                dta['Holding'][x] = dta['Total Assets'][x - 1] // dta['Close'][x]
                dta['Employed Capital'][x] = dta['Holding'][x] * dta['Close'][x]
                dta['Free Capital'][x] = dta['Total Assets'][x -1] - dta['Employed Capital'][x]
                dta['Total Assets'][x] = dta['Free Capital'][x] + dta['Employed Capital'][x]
            else:
                dta['Employed Capital'][x] = dta['Employed Capital'][x - 1]
                profit_loss = (dta['Holding'][x - 1] * dta['Close'][x]) - shorted_cap
                dta['Total Assets'][x] = dta['Free Capital'][x - 1] + dta['Employed Capital'][x] + profit_loss
                dta['Free Capital'][x] = dta['Total Assets'][x]
                dta['Holding'][x] = 0
                dta['Shorted Capital'][x] = 0
#                 print(profit_loss)
        elif dta['Position'][x] == -1:
            if dta['Holding'][x - 1] == 0:
                dta['Holding'][x] = -(dta['Total Assets'][x - 1] // dta['Close'][x])
                dta['Employed Capital'][x] = dta['Employed Capital'][x - 1]
                dta['Free Capital'][x] = dta['Free Capital'][x - 1]
                shorted_cap = dta['Holding'][x] * dta['Close'][x]
#                 print(shorted_cap)
                dta['Shorted Capital'][x] = dta['Holding'][x] * dta['Close'][x]
                dta['Total Assets'][x] = dta['Free Capital'][x] + dta['Employed Capital'][x]
            else:
                dta['Shorted Capital'][x] = dta['Shorted Capital'][x - 1]
                dta['Free Capital'][x] = dta['Free Capital'][x - 1]
                dta['Employed Capital'][x] = 0
                dta['Total Assets'][x] = dta['Free Capital'][x] + dta['Holding'][x - 1] * dta['Close'][x]
                
        else:
            dta['Holding'][x] = dta['Holding'][x - 1]
            if dta['Holding'][x] > 0:
                dta['Shorted Capital'][x] = dta['Shorted Capital'][x - 1]
                dta['Free Capital'][x] = dta['Free Capital'][x - 1]
                dta['Employed Capital'][x] = dta['Holding'][x] * dta['Close'][x]
                dta['Total Assets'][x] = dta['Free Capital'][x] + dta['Employed Capital'][x]
            elif dta['Holding'][x] < 0:
                dta['Employed Capital'][x] = dta['Employed Capital'][x - 1]
                dta['Free Capital'][x] = dta['Free Capital'][x - 1]
                dta['Shorted Capital'][x] = dta['Holding'][x] * dta['Close'][x]
                dta['Total Assets'][x] = dta['Free Capital'][x] + dta['Employed Capital'][x] 
            else:
                dta['Free Capital'][x] = dta['Free Capital'][x - 1]
                dta['Employed Capital'][x] = dta['Employed Capital'][x - 1]
                dta['Shorted Capital'][x] = dta['Shorted Capital'][x - 1]
                dta['Total Assets'][x] = dta['Total Assets'][x - 1]
    if dta['Holding'][len(dta) - 1] > 0:
        profit = (dta['Total Assets'][len(dta) - 1] / dta['Total Assets'][0] - 1) * 100
    elif dta['Holding'][len(dta) - 1] < 0:
        profit1 = dta['Shorted Capital'][len(dta) - 1] - shorted_cap
        profit = ((profit1 + dta['Total Assets'][len(dta) - 1]) / dta['Total Assets'][0] - 1) * 100
    else:
        profit = (dta['Total Assets'][len(dta) - 1] / dta['Total Assets'][0] - 1) * 100
    return(profit)

# In[10]:

def distance_method(date1, date2, date3, date4, x, df, capital=10000): # formation period: date1, date2. trading period: date3, date4
    coeff_matrix, coeff_list = forming_pairs(date1, date2, df)
    pairs = pair_tickers(coeff_matrix, coeff_list, x)
    print(pairs)
    period_return = []
    for i in pairs:
        pair1 = i[0]
        pair2 = i[1]
        df_pair1, df_pair2 = trading_df(pair1, pair2, date1, date2, date3, date4, df)
        profit_1 = trading(df_pair1, capital)
        profit_2 = trading(df_pair2, capital)
        pair_return = pair_profit(profit_1, profit_2)
        period_return.append(pair_return)
    pairs_and_returns = dict(zip(period_return, pairs)) 
    return(period_return)
    return(pairs)

# =============================================================================
# def distance_method(date1, date2, date3, date4, n, df, capital=10000):
#     coeff_matrix, coeff_list = forming_pairs(date1, date2, df)
#     pairs = pair_tickers(coeff_matrix, coeff_list, n, df, date1, date2, date3, date4)
#     print(pairs)
#     period_return = []
#     for pair in pairs:
#         df_pair1, df_pair2 = trading_df(pair[0], pair[1], date1, date2, date3, date4, df)
#         profit_1 = trading(df_pair1, capital)
#         profit_2 = trading(df_pair2, capital)
#         pair_return = pair_profit(profit_1, profit_2)
#         period_return.append(pair_return)
#     return period_return
# =============================================================================

# In[11]:

def average(sequence): # input a list
    sum_ = sum(sequence)
    mean = sum_ / len(sequence)
    return(mean)

#%% Plot the pairs with std

def cum_return(df):
    cumret = np.log(df).diff().cumsum()+1
    return cumret

def calculate_distances(df):
    '''
    calculate Euclidean distance for each pair of stocks in the dataframe
    return sorted dictionary (in ascending order)
    '''
    cumret = cum_return(df)
    distances = {} # dictionary with distance for each pair
    
    # calculate distances
    for s1 in cumret.columns:
        for s2 in cumret.columns:
            if s1!=s2 and (f'{s1}-{s2}' not in distances.keys()) and (f'{s2}-{s1}' not in distances.keys()):
                dist = np.sqrt(np.sum((cumret[s1] - cumret[s2])**2)) # Euclidean distance
                distances[f'{s1}-{s2}'] = dist
    
    # sort dictionary
    sorted_distances = {k:v for k,v in sorted(distances.items(), key = lambda item: item[1])}
    
    return sorted_distances

def parse_pair(pair):
    '''
    parse pair string S1-S2
    return tickers S1, S2
    '''
    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp+1:]
    
    return s1,s2

def cum_return_train(df, date1, date2):
    cumret_train = cum_return(df)
    cumret_train = cumret_train.loc[date1:]
    cumret_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    cumret_train.dropna(axis=1,inplace=True)
    cumret_train = cumret_train / cumret_train.iloc[0] # divide by first row so that all prices start at 1 
    cumret_train = cumret_train.loc[date1:date2]
    return cumret_train

def cum_return_test(df, date1, date2, date3, date4):
    cumret_test = cum_return(df)
    cumret_test = cumret_test.loc[date1:]
    cumret_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    cumret_test.dropna(axis=1,inplace=True)
    cumret_test = cumret_test / cumret_test.iloc[0] # divide by first row so that all prices start at 1
    cumret_test = cumret_test.loc[date3:date4]
    return cumret_test

#%%
import matplotlib.pyplot as plt

def plot_pairs_distance(df, date1, date2, date3, date4, N=1):
    '''
    plot cumulative returns of the spread for each of N pairs with smallest Euclidean distance
    '''
    cumret_train = cum_return_train(df, date1, date2)
    cumret_test = cum_return_test(df, date1, date2, date3, date4)
    sorted_distances = calculate_distances(cumret_train)
    pairs = [k for k,v in sorted_distances.items()][:N]
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        spread_train = cumret_train[s1] - cumret_train[s2]
        spread_test = cumret_test[s1] - cumret_test[s2]
        spread_mean = spread_train.mean() # historical mean
        spread_std = spread_train.std() # historical standard deviation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))
        fig.suptitle(f'Spread of {pair} pair', fontsize=16)
        ax1.plot(spread_train, label='spread')
        ax1.set_title('Formation period')
        ax1.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax1.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax1.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax1.legend()
        ax2.plot(spread_test, label='spread')
        ax2.set_title('Trading period')
        ax2.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax2.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax2.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax2.legend()

#%%

def calculate_metrics_distance(df, date1, date2, date3, date4, N=5):
    '''
    calculate metrics for N pairs with the smallest Euclidean distance
    return dataframe of results
    '''
    cumret_train = cum_return_train(df, date1, date2)
    cumret_test = cum_return_test(df, date1, date2, date3, date4)
    sorted_distances = calculate_distances(cumret_train)
    pairs = [k for k,v in sorted_distances.items()][:N]
    
    cols = ['Euclidean distance', 'Spread SD', 
            'Num zero-crossings', '% days within 2-SD band']
    results_formation = pd.DataFrame(index=pairs, columns=cols)
    results_trading = pd.DataFrame(index=pairs, columns=cols)
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        spread = cumret_train[s1] - cumret_train[s2]
        results_formation.loc[pair]['Euclidean distance'] = np.sqrt(np.sum((spread)**2))
        results_formation.loc[pair]['Spread SD'] = spread.std()
        results_formation.loc[pair]['Num zero-crossings'] = ((spread[1:].values * spread[:-1].values) < 0).sum()
        results_formation.loc[pair]['% days within 2-SD band'] = (abs(spread) < 2*spread.std()).sum() / len(spread) * 100
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        spread = cumret_test[s1] - cumret_test[s2]
        results_trading.loc[pair]['Euclidean distance'] = np.sqrt(np.sum((spread)**2))
        results_trading.loc[pair]['Spread SD'] = spread.std()
        results_trading.loc[pair]['Num zero-crossings'] = ((spread[1:].values * spread[:-1].values) < 0).sum()
        results_trading.loc[pair]['% days within 2-SD band'] = (abs(spread) < 2*spread.std()).sum() / len(spread) * 100
        
    return results_formation, results_trading


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module used to store the functions to implement:
    - cointegration approach inspired by 
    Engle and Granger, 1987 "Co-Integration and Error Correction: Representation, Estimation, and Testing."
    
"""

#%% Import libraries

import pandas as pd
import numpy as np
import statistics as stat
# import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

# In[2]: Function to find cointegrated pairs

def find_cointegrated_pairs(date1, date2, df):
    data = df.loc[date1 : date2]
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append([keys[i], keys[j]])
    return score_matrix, pvalue_matrix, pairs

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

def pair_tickers(coeff_matrix, coeff_list):
    pair_names = []
    for c in coeff_list:
        for i in coeff_matrix.index:
            for j in coeff_matrix.columns:
                if coeff_matrix.loc[i,j] == c:
                    names = [i,j]
                    pair_names.append(names)
    for i in pair_names:
        pair_names.remove([i[1], i[0]])
    return(pair_names)

def pair_profit(profit1, profit2):
    profit = (profit1 + profit2) / 2
    return(profit)

#%% Functions for operation of cointegration approach

# obtain list of coin pairs
def coint_pairs(date1, date2, x, df):
    sd_mat, sd_coef = forming_pairs(date1, date2, df)
    sd_pairs = pair_tickers(sd_mat, sd_coef)
    _, _, coint_pairs = find_cointegrated_pairs(date1, date2, df)
    pairs = []
    for i in sd_pairs:
        pair1 = i[0]
        pair2 = i[1]
        if [pair1, pair2] in coint_pairs or [pair2, pair1] in coint_pairs: 
            pairs.append(i)
        else:
            pass
    return(pairs[:x])

#%% calculate spread after regression

def spread_stat(date1, date2, pair1, pair2, df): # dates are for the formation period
    dta = df.loc[date1 : date2]
    data = dta.divide(dta.loc[date1])
    s1 = data[pair1]
    s2 = data[pair2]
    
    s1 = sm.add_constant(s1)
    results = sm.OLS(s2, s1).fit()
    s1 = s1[pair1]
    b = results.params[pair1]
    
    spread = s2 - b * s1
    std = spread.std()
    mean = spread.mean()
    return(std, mean)

# In[7]: Trading Signals

def signal(date1, date2, std, mean, pair1, pair2, df):
    dta = df.loc[date1 : date2]
    data = dta.divide(dta.loc[date1])
    s1 = data[pair1]
    s2 = data[pair2]
    
    s1 = sm.add_constant(s1)
    results = sm.OLS(s2, s1).fit()
    s1 = s1[pair1]
    b = results.params[pair1]
    assert(len(s1) == len(s2))
    spread = s2 - b * s1
    norm_spread = (spread - mean) / std
    df1 = pd.DataFrame(index = s1.index, columns = ['Close', 'norm_spread'])
    df1['Close'] = dta[pair1]
    df1['norm_spread'] = norm_spread
    df2 = pd.DataFrame(index = s2.index, columns = ['Close', 'norm_spread'])
    df2['Close'] = dta[pair2]
    df2['norm_spread'] = norm_spread
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df1['Signal'] = pd.Series(np.zeros(len(df1)))
    df1['Signal'][0] = 0
    df2['Signal'] = pd.Series(np.zeros(len(df2)))
    df2['Signal'][0] = 0
    for x in range(1, len(df1)):
        if df1['norm_spread'][x] > 2:
            df1['Signal'][x] = 1
            df2['Signal'][x] = -1
        elif df1['norm_spread'][x] < -2:
            df1['Signal'][x] = -1
            df2['Signal'][x] = 1
        elif df1['norm_spread'][x] == 0:
            df1['Signal'][x] = 0
            df2['Signal'][x] = 0
        elif (df1['norm_spread'][x - 1]) / (df1['norm_spread'][x]) < 0:
            df1['Signal'][x] = 0
            df2['Signal'][x] = 0
        else:
            df1['Signal'][x] = df1['Signal'][x - 1]
            df2['Signal'][x] = df2['Signal'][x - 1]
    df1['Position'] = pd.Series(np.zeros(len(df1)))
    df1['Position'] = df1['Signal'].diff()
    df2['Position'] = pd.Series(np.zeros(len(df2)))
    df2['Position'] = df2['Signal'].diff()
    return(df1, df2)

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

#%% Main function for cointegration approach

def Coint_Trading(date1, date2, date3, date4, n, df, capital=10000): # date1, date2 formation period, date3, date4 trading period, x stocks
    pairs = coint_pairs(date1, date2, n, df)
    print(pairs)
    returns = []
    for i in pairs:
        pair1 = i[0]
        pair2 = i[1]
        std, mean = spread_stat(date1, date2, pair1, pair2, df)
        df1, df2 = signal(date3, date4, std, mean, pair1, pair2, df)
        pair1_r = trading(df1, capital)
        pair2_r = trading(df2, capital)
        pair_return = pair_profit(pair1_r, pair2_r)
        returns.append(pair_return)
    return(returns)
    return(pairs)

#%% Mean for monthly return

def average(sequence): # input a list
    sum_ = sum(sequence)
    mean = sum_ / len(sequence)
    return(mean)

#%% Plot the pairs with std

def cum_return(df):
    cumret = np.log(df).diff().cumsum()+1
    return cumret

def parse_pair(pair):
    '''
    parse pair string S1-S2
    return tickers S1, S2
    '''
    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp+1:]
    
    return s1,s2

def cadf_pvalue(s1, s2, cumret):
    '''
    perform CADF cointegration tests
    since it is sensitive to the order of stocks in the pair, perform both tests (s1-2 and s2-s1)
    return the smallest p-value of two tests
    '''
    from statsmodels.tsa.stattools import coint
    
    p1 = coint(cumret[s1], cumret[s2])[1]
    p2 = coint(cumret[s2], cumret[s1])[1]
    
    return min(p1,p2)

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

def select_pairs(train):
    '''
    select pairs using data from train dataframe
    return dataframe of selected pairs
    '''
    tested = []

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
# =============================================================================
#     from hurst import compute_Hc
#     from statsmodels.tsa.stattools import adfuller
# =============================================================================
    from statsmodels.tsa.stattools import coint

    cols = ['Distance', 'Num zero-crossings', 'Pearson r', 'Spread mean', 
            'Spread SD', 'Hedge ratio']
    pairs = pd.DataFrame(columns=cols)

    for s1 in train.columns:
        for s2 in train.columns:
            if s1!=s2 and (f'{s1}-{s2}' not in tested):
                tested.append(f'{s1}-{s2}')
                cadf_p = coint(train[s1], train[s2])[1]
                if cadf_p<0.01 and (f'{s2}-{s1}' not in pairs.index): # stop if pair already added as s2-s1
                    res = OLS(train[s1], add_constant(train[s2])).fit()
                    hedge_ratio = res.params[s2]
                    if hedge_ratio > 0: # hedge ratio should be posititve
                        spread = train[s1] - hedge_ratio*train[s2]
# =============================================================================
#                         hurst = compute_Hc(spread)[0]
#                         if hurst<0.5:
#                             halflife = calculate_halflife(spread)
#                             if halflife>1 and halflife<30:
# =============================================================================
                        # subtract the mean to calculate distances and num_crossings
                        spread_nm = spread - spread.mean() 
                        num_crossings = (spread_nm.values[1:] * spread_nm.values[:-1] < 0).sum()
                        if num_crossings>len(train.index)/252*12: 
                            distance = np.sqrt(np.sum(spread_nm**2))
                            pearson_r = np.corrcoef(train[s1], train[s2])[0][1]
                            pairs.loc[f'{s1}-{s2}'] = [distance, num_crossings, pearson_r, spread.mean(),
                                                       spread.std(), hedge_ratio]
                                
    return pairs

#%%

import matplotlib.pyplot as plt

def plot_pairs_cointegration(df, date1, date2, date3, date4, N=1):
    '''
    plot cumulative returns of the spread for each pair in pairs
    '''
    
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    cumret_train = cum_return_train(df, date1, date2)
    cumret_test = cum_return_test(df, date1, date2, date3, date4)
    pairs = select_pairs(cumret_train)
    pairs = list(pairs.sort_values(by='Distance').index[:N])
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        res = OLS(cumret_train[s1], add_constant(cumret_train[s2])).fit()
        spread_train = cumret_train[s1] - res.params[s2]*cumret_train[s2]
        spread_test = cumret_test[s1] - res.params[s2]*cumret_test[s2]
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

def calculate_metrics_cointegration(df, date1, date2, date3, date4, N=1):
    '''
    calculate metrics for pairs using data in cumret
    return dataframe of results
    '''
    # from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import coint
    
    cumret_train = cum_return_train(df, date1, date2)
    cumret_test = cum_return_test(df, date1, date2, date3, date4)
    pairs_df = select_pairs(cumret_train)
    pairs = list(pairs_df.sort_values(by='Distance').index[:N])
    
    cols = ['Distance', 'CADF p-value', 'ADF p-value', 'Spread SD', 'Pearson r',
        'Num zero-crossings', '% days within historical 2-SD band']
    results_formation = pd.DataFrame(index=pairs, columns=cols)
    results_trading = pd.DataFrame(index=pairs, columns=cols)
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
        spread = cumret_train[s1] - hedge_ratio*cumret_train[s2]
        results_formation.loc[pair]['CADF p-value'] = coint(cumret_train[s1], cumret_train[s2])[1]
        results_formation.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        hist_mu = pairs_df.loc[pair]['Spread mean'] # historical mean
        hist_sd = pairs_df.loc[pair]['Spread SD'] # historical standard deviation
        results_formation.loc[pair]['Spread SD'] = hist_sd
        results_formation.loc[pair]['Pearson r'] = np.corrcoef(cumret_train[s1], cumret_train[s2])[0][1]
        # subtract the mean to calculate distances and num_crossings
        spread_nm = spread - hist_mu
        results_formation.loc[pair]['Distance'] = np.sqrt(np.sum((spread_nm)**2))
        results_formation.loc[pair]['Num zero-crossings'] = ((spread_nm[1:].values * spread_nm[:-1].values) < 0).sum()
        # results.loc[pair]['Hurst Exponent'] = compute_Hc(spread)[0]
        # results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results_formation.loc[pair]['% days within historical 2-SD band'] = (abs(spread-hist_mu) < 2*hist_sd).sum() / len(spread) * 100

    for pair in pairs:
        s1,s2 = parse_pair(pair)
        hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
        spread = cumret_test[s1] - hedge_ratio*cumret_test[s2]
        results_trading.loc[pair]['CADF p-value'] = coint(cumret_test[s1], cumret_test[s2])[1]
        results_trading.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        hist_mu = pairs_df.loc[pair]['Spread mean'] # historical mean
        hist_sd = pairs_df.loc[pair]['Spread SD'] # historical standard deviation
        results_trading.loc[pair]['Spread SD'] = hist_sd
        results_trading.loc[pair]['Pearson r'] = np.corrcoef(cumret_test[s1], cumret_test[s2])[0][1]
        # subtract the mean to calculate distances and num_crossings
        spread_nm = spread - hist_mu
        results_trading.loc[pair]['Distance'] = np.sqrt(np.sum((spread_nm)**2))
        results_trading.loc[pair]['Num zero-crossings'] = ((spread_nm[1:].values * spread_nm[:-1].values) < 0).sum()
        # results.loc[pair]['Hurst Exponent'] = compute_Hc(spread)[0]
        # results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results_formation.loc[pair]['% days within historical 2-SD band'] = (abs(spread-hist_mu) < 2*hist_sd).sum() / len(spread) * 100
        
    return results_formation, results_trading
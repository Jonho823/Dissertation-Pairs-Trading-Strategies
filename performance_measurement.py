# -*- coding: utf-8 -*-
"""
This module leverages the QuantStats library to calculate performance metrics for pairs trading strategies. 

This module provides functions to analyze the returns and presents the performance metrics in DataFrame format. 
"""
#%%

# import libraries

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import quantstats as qs
import sys
import io

qs.extend_pandas()

#%%

def performance_report(input_path, rf=0.02, display=False):
    # Read files
    returns = pd.read_excel(input_path)
    
    # Drop any na rows
    amended_returns = returns.dropna()
    
    # Drop extra date columns (date 2, date 3, date 4)
    amended_returns.drop(['date2', 'date3', 'date4'], axis=1, inplace=True)

    # Convert the 'date 1' column to datetime format
    amended_returns['date1'] = pd.to_datetime(amended_returns['date1'])

    # Set the 'date 1' column as the index
    amended_returns.set_index('date1', inplace=True)

    # Turn returns into pct by div 100
    amended_returns['returns'] = amended_returns['returns'] / 100

    # Convert dataframe to series
    returns_ser = amended_returns['returns'].squeeze()
    
    # Redirect stdout to capture printed output
    sys.stdout = io.StringIO()
    
    # Run qs library and extract results
    qs.reports.metrics(returns_ser, rf=rf)
    # qs.reports.html(returns_ser, title='top20_Coint_pCOVID', output=output_html)
    
    # Get the printed output
    printed_output = sys.stdout.getvalue()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Parse the printed output into a DataFrame
    report_metrics = pd.read_csv(io.StringIO(printed_output), delimiter="\s\s+", engine="python")[1:]

    return report_metrics

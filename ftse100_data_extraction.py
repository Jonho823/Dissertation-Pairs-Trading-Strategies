#

#%% Import libraries

import bs4 as bs
import requests # get the HTML data from the website
import pandas as pd
import datetime as dt
import yfinance as yf

#%%

def get_ftse100_symbols():
    """
    Function to retrieve the symbols of companies in the FTSE 100 index from Wikipedia.
    
    Code reference to Youtube: NeuralNine_S&P 500 Web Scraping with Python
    
    Step 1: 
        Figure the structure of the website
        What the HTML code looks like
        Right click the table > Inspect element
        
    Step 2:
        Send an HTTP requests to the website
        Get the HTML code as response
        
    Step 3:
        HTML code fed into a soup object
        Soup object scrape through the HTML code
        > filter out the data we are interested
    
    Returns:
    symbols (list): A list of company symbols in the FTSE 100 index.
    """
    # Get the HTML code
    html = requests.get('https://en.wikipedia.org/wiki/FTSE_100_Index#Current_constituents')
    
    # Use a soup object to optimize the process to filter the data
    soup = bs.BeautifulSoup(html.text, 'html.parser')
    
    # Define a ticker list to store the tickers
    symbols = []
    
    # Find the first table elements
    table = soup.find('table', {'id': 'constituents'})
    
    # Get all the rows of the table and get the first column for each row
    rows = table.find_all('tr')[1:] # find all data except 1st row
    
    # Use for loop to extract symbols to tickers list
    for row in rows:
        symbol = row.find_all('td')[1].text + '.L'
        symbols.append(symbol)
    
    return symbols

#%%

def download_stock_data(symbols, start_date, end_date, output_file):
    """
    Function to download stock data from Yahoo Finance using yfinance library.

    Parameters:
    symbols (list): List of stock symbols.
    start_date (str or datetime): Start date for the data download in 'YYYY-MM-DD' format or as datetime object.
    end_date (str or datetime): End date for the data download in 'YYYY-MM-DD' format or as datetime object.
    output_file (str): File path to save the downloaded data.

    Returns:
    1. A csv file contains daily adj close price of list of stock(s).
    2. Identify any na columns and display it.
    """
    # If we have a datareader (web), but should not be web
    # Below enable Yahoo Finance's Pandas DataReader (PDR) override for compatibility with yfinance.
    # This is likely related to fetching data from Yahoo Finance
    yf.pdr_override()

    # Convert start_date and end_date to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

    # Create a dataframe for storing prices
    prices = pd.DataFrame()

    # Download data for each symbol
    for symbol in symbols:
        tmpdf = yf.download(symbol, start_date, end_date)
        prices[symbol] = tmpdf['Adj Close']

    # Fill NaN values with 0
    prices = prices.fillna(0)

    # Save prices to file
    prices.to_csv(output_file)

    # Print columns with NA values if any
    na_columns = prices.columns[prices.isna().any()]
    if len(na_columns) > 0:
        print("Columns with NA values:", na_columns)
    else:
        print("No columns with NA values.")



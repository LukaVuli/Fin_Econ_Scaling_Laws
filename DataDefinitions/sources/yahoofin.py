import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import yfinance as yf
import datetime

def available_data():
    y_fin_data = ['TICKER_X{period}{}_Y{interval}{}', 'SPY_2y_1d']
    return y_fin_data

def yfindata(item, per, inter):
    df_yahoo = 5 #TODO: replace this with what I have in scratch.py I like that method better than what I had before here.
    return df_yahoo


if __name__ == "__main__":
    print('Hello World')
    #TODO: make sure to build this out. ITEM is TICKER_PERIOD_INTERVAL in the dd, it will be seperated in dd so yfindata can be called.

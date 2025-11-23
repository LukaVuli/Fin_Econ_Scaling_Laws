import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd



def get_portfolio_returns(threshold = 0.1, share_code = [10, 11], exchangecode = [1, 2, 4], long_over_short = True):
    #TODO: I want to query the data from WRDS, then do the computation :)
    print('Hello World')
# Get ratio data
file_path = '/Users/lukavulicevic/Desktop/assignment2data.csv'
df = pd.read_csv(file_path, low_memory=False)
financial_ratio = 'adv_sale'
columns_to_keep = ['cusip', financial_ratio, 'qdate', 'public_date']
ratios = df[columns_to_keep]
ratios['public_date'] = pd.to_datetime(ratios['public_date'])  # Ensure it's in datetime format

# Get return data
file_path = '/Users/lukavulicevic/Desktop/assignment2returns.csv'
returns = pd.read_csv(file_path, low_memory=False)
returns.rename(columns={'CUSIP': 'cusip', 'date': 'public_date'}, inplace=True)
returns['public_date'] = pd.to_datetime(returns['public_date'])  # Ensure it's in datetime format
returns['public_date'] = returns['public_date'] + pd.offsets.MonthEnd(0)

#
#
#
#Part 1
#
#
#
# Merge the ratios with returns on 'cusip' and 'public_date'
merged_df = pd.merge(ratios, returns, on=['cusip', 'public_date'], how='inner')
# Remove duplicates and rows with missing entries
merged_df.drop_duplicates(inplace=True)
merged_df.dropna(inplace=True)

# Calculate market cap and set index
merged_df['MKTCAP'] = merged_df['PRC'] * merged_df['SHROUT']
merged_df['public_date'] = pd.to_datetime(merged_df['public_date'])  # Convert to datetime
merged_df.set_index(['cusip', 'public_date'], inplace=True)

# Resample to end of quarter data to pick the portfolio stocks at the end of each quarter
quarterly = merged_df.resample('Q', level='public_date')

zero_count = (merged_df[financial_ratio] == 0).sum()

# Create a function to get the high and low portfolios based on adv_sale ratio
def get_high_low_portfolios(df):
    df = df.dropna(subset=[financial_ratio])
    # Find the threshold for the top 10% and bottom 10%
    high_thresh = df[financial_ratio].quantile(0.9)
    low_thresh = df[financial_ratio].quantile(0.1)

    #print("HIGH: " + str(high_thresh) + ", LOW: " + str(low_thresh))

    # Get high and low portfolios
    high_portfolio = df[df[financial_ratio] >= high_thresh]
    low_portfolio = df[df[financial_ratio] <= low_thresh]

    return high_portfolio, low_portfolio


# Dictionary to store portfolio weights and returns
high_weights = {}
low_weights = {}
portfolio_returns = []

# Loop through each quarter
for quarter_end, quarter_df in quarterly:
    high_portfolio, low_portfolio = get_high_low_portfolios(quarter_df)

    # Compute value weights for the high and low portfolios
    high_portfolio['weight'] = high_portfolio['MKTCAP'] / high_portfolio['MKTCAP'].sum()
    low_portfolio['weight'] = low_portfolio['MKTCAP'] / low_portfolio['MKTCAP'].sum()

    # Store the weights for use in the next quarter
    high_weights[quarter_end] = high_portfolio[['weight']]
    low_weights[quarter_end] = low_portfolio[['weight']]

    # Get returns for the next quarter
    next_quarter = pd.date_range(quarter_end + pd.DateOffset(days=1), periods=3, freq='M')

    for month in next_quarter:
        # Ensure only the stocks that were in the portfolio at the end of the quarter are used
        month_norm = pd.to_datetime(month).normalize()

        high_month_returns = merged_df.loc[
            (merged_df.index.get_level_values('public_date').normalize() == month_norm) &
            (merged_df.index.get_level_values('cusip').isin(high_portfolio.index.get_level_values('cusip'))),
            'RET'
        ]

        low_month_returns = merged_df.loc[
            (merged_df.index.get_level_values('public_date').normalize() == month_norm) &
            (merged_df.index.get_level_values('cusip').isin(low_portfolio.index.get_level_values('cusip'))),
            'RET'
        ]

        # Apply weights to compute weighted returns
        high_month_returns_reset = high_month_returns.reset_index(level='cusip')
        high_portfolio_reset = high_portfolio.reset_index(level='cusip')
        high_month_returns = pd.merge(high_month_returns_reset, high_portfolio_reset[['weight', 'cusip']], how="inner", on='cusip')
        high_month_returns['RET'] = pd.to_numeric(high_month_returns['RET'], errors='coerce')
        high_month_returns['weight'] = pd.to_numeric(high_month_returns['weight'], errors='coerce')

        high_return = (high_month_returns['RET'] * high_month_returns['weight']).sum()

        low_month_returns_reset = low_month_returns.reset_index(level='cusip')
        low_portfolio_reset = low_portfolio.reset_index(level='cusip')
        low_month_returns = pd.merge(low_month_returns_reset, low_portfolio_reset[['weight', 'cusip']], how="inner", on='cusip')
        low_month_returns['RET'] = pd.to_numeric(low_month_returns['RET'], errors='coerce')
        low_month_returns['weight'] = pd.to_numeric(low_month_returns['weight'], errors='coerce')

        low_return = (low_month_returns['RET'] * low_month_returns['weight']).sum()

        # Store monthly portfolio returns
        portfolio_returns.append({
            'date': month,
            'high_return': high_return,
            'low_return': low_return,
            financial_ratio: high_return - low_return
        })

# Convert the results to a DataFrame
new_factor = pd.DataFrame(portfolio_returns)
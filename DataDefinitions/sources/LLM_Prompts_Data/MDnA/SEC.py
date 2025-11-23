import dropbox
import pandas as pd
import io
from credentials import dropbox_token, first_name, last_name, my_email
import os

def available_data():
    arch = ['']
    return arch


def SEC_TICKERS():
    try:
        # get the data:
        sec_data_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": first_name + ' ' + last_name + ' ' + my_email  # Replace with your name and email
        }
        response = requests.get(sec_data_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame.from_dict(data, orient='index')
        else:
            print(f"Failed to fetch data: {response.status_code}")
            df = None
    except Exception as err:
        print(f"An error occurred: {err}")
        df = None
    return df


if __name__ == "__main__":
    analysis_round = 1
    if analysis_round == 1:
        sec_data_url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": "Luka Vuliceivc lvulicevic@ucsd.edu"  # Replace with your name and email
        }
        response = requests.get(sec_data_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            companies = pd.DataFrame.from_dict(data, orient='index')
            print(companies.head())
        else:
            print(f"Failed to fetch data: {response.status_code}")
        sampled_companies = companies.sample(n=1000, random_state=42)
        print(sampled_companies.head())
        sampled_companies.to_csv("/Users/lukavulicevic/Desktop/sampled_companies.csv", index=False)
        returns_data = {}
        for _, row in sampled_companies.iterrows():
            ticker = row["ticker"]  # Assuming 'ticker' column exists
            print(f"Fetching data for {ticker}...")
            returns_data[ticker] = fetch_daily_data(ticker)

        start_date = "2000-01-01"
        end_date = "2015-12-31"
        min_days = 252 * 2
        combined_returns = pd.DataFrame()
        for ticker, df in returns_data.items():
            if df is not None and "Returns" in df.columns:
                df = df.loc[start_date:end_date]
                if df["Returns"].notna().sum() >= min_days:
                    combined_returns = pd.concat([combined_returns, df["Returns"].rename(ticker)], axis=1)

        valid_tickers = combined_returns.columns.tolist()
        filtered_sampled_companies = sampled_companies[sampled_companies["ticker"].isin(valid_tickers)]
        filtered_sampled_companies = filtered_sampled_companies.copy()
        filtered_sampled_companies["year"] = np.nan
        for idx, row in filtered_sampled_companies.iterrows():
            ticker = row["ticker"]
            valid_dates = combined_returns[ticker].dropna().index
            valid_years = pd.to_datetime(valid_dates).year.unique()
            if len(valid_years) > 0:
                sampled_year = np.random.choice(valid_years)
                filtered_sampled_companies.at[idx, "year"] = sampled_year
        filtered_sampled_companies["year"] = filtered_sampled_companies["year"].astype(int)



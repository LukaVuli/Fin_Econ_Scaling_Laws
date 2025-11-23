import dropbox
import pandas as pd
import io
from credentials import dropbox_token

def available_data():
    arch = ['q-factors(2023-12)']
    return arch


def qfac(item, starting, ending):
    FILE_PATH = '/q-factor/' + item + '.csv'
    try:
        dbx = dropbox.Dropbox(dropbox_token)
        _, res = dbx.files_download(FILE_PATH)
        df = pd.read_csv(io.BytesIO(res.content))
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df.set_index('date', inplace=True)
        df.drop(columns=['year', 'month'], inplace=True)
        df = df/100
        if starting == None:
            starting = df.index[0]
        if ending == None:
            ending = df.index[-1]
        start_date = pd.to_datetime(starting)
        end_date = pd.to_datetime(ending)
        df = df.loc[start_date:end_date]

        df = df.rename(columns={'R_ME': 'SMB'})
        df = df.rename(columns={'R_IA': 'IA'})
        df = df.rename(columns={'R_ROE': 'ROE'})
        df = df.rename(columns={'R_EG': 'EG'})
        df = df.rename(columns={'R_F': 'RF'})
        df = df.rename(columns={'R_MKT': 'Mkt-RF'})

    except dropbox.exceptions.AuthError as err:
        print(f"Authentication error: {err}")
        df = None
    except dropbox.exceptions.ApiError as err:
        print(f"API error: {err}")
        df = None
    except Exception as err:
        print(f"An error occurred: {err}")
        df = None
    return df

if __name__ == "__main__":
    print('Hello World')
    qfac('q-factors(2023-12)', '1900-01-01', '2024-01-01')
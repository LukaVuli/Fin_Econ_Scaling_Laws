import dropbox
import pandas as pd
import io
from credentials import dropbox_token

def available_data():
    arch = ['']
    return arch


def etfdata(item, starting, ending):

    FILE_PATH = '/ETF Intraday/' + item[0] + '/' + item + '_full_1min_adjsplitdiv' + '.txt'
    column_names = ['date', 'open', 'high', 'low', 'close', 'volume']

    try:
        dbx = dropbox.Dropbox(dropbox_token)
        _, res = dbx.files_download(FILE_PATH)

        # Read the TXT file content
        df = pd.read_csv(io.BytesIO(res.content), sep=',', names=column_names)  # Assuming tab-separated values
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        if starting is None:
            starting = df.index[0]
        if ending is None:
            ending = df.index[-1]
        start_date = pd.to_datetime(starting)
        end_date = pd.to_datetime(ending)
        df = df.loc[start_date:end_date]
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
    etfdata('AAA', '1900-01-01', '2024-01-01')
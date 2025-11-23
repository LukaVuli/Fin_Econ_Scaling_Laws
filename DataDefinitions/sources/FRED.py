import warnings
warnings.filterwarnings('ignore')
import pandas_datareader.data as web

def available_data():
    fred_data = ['GDP'] #TODO: find more series???
    return fred_data

def fred(data,starting,ending):
    return web.DataReader(data, 'fred', starting, ending)


if __name__ == "__main__":
    print('Hello World')
import pandas as pd
from DataDefinitions.sources import famafrench as ff
from DataDefinitions.sources import FRED as FRED
from DataDefinitions.sources import q_factors as QF
from DataDefinitions.sources import ETF
from DataDefinitions.sources import yahoofin as yfin
import datetime
class dd():
    def __init__(self, source, item, start, end):
        self.start = start
        self.end = end
        self.source = source
        self.item = item

        if source == 'famafrench':
            if self.item == None:
                self.data = ff.available_data()
            else:
                self.data = ff.data_getter(self.item, self.start, self.end)
        elif source == 'fred':
            if self.item == None:
                self.data = FRED.available_data()
            else:
                self.data = FRED.fred(self.item, self.start, self.end)
        elif source == 'qfac':
            if self.item == None:
                self.data = QF.available_data()
            else:
                self.data = QF.qfac(self.item, self.start, self.end)
        elif source == 'ETF':
            if self.item == None:
                self.data = ETF.available_data()
            else:
                self.data = ETF.etfdata(self.item, self.start, self.end)
        elif source == 'yfin':
            if self.item == None:
                self.data = yfin.available_data()
            else:
                #TODO I need to seperate item by '_'. First part is ticker, next part is period, last part is interval.
                self.data = yfin.yfindata()
        else:
            self.data = 'You have selected an improper source. Try again.'
            self.start = None
            self.end = None
            self.item = None

    def extract(self):
        return self.data


if __name__ == "__main__":
    ME_ports = dd('famafrench', item='F-F_Research_Data_5_Factors_2x3_daily', start='1900-01-01', end=None)
    ME_ports = dd('famafrench', item='F-F_Research_Data_5_Factors_2x3_daily', start='1900-01-01', end=None)

    data = ME_ports.extract()
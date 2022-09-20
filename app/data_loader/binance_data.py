import os

import pandas as pd
from binance import Client

API_KEY = os.environ.get('API_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')


class BinanceDataLoader:

    def __init__(self):
        self.client = Client(API_KEY, SECRET_KEY)

    def get_data(self, asset_name: str):
        columns = ['Kline Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price',
                   'Volume', 'Kline Close Time', 'Quote Asset Volume', 'Number of Trades',
                   'Taker Buy ' 'Base Asset ' 'Volume', 'Taker Buy Quote Asset Volume', 'Unused']
        res = pd.DataFrame(self.client.get_historical_klines(asset_name,
                                                             Client.KLINE_INTERVAL_30MINUTE,
                                                             "1 Jan, "
                                                             "2017",
                                                             "1 Jan, 2022"), columns=columns)

        return res

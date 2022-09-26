import datetime as dt
import os

import pandas as pd
from binance import Client, enums

API_KEY = os.environ.get("API_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")


def fix_df(df):
    # make them numbers
    df = df.apply(pd.to_numeric, errors="coerce")
    # get the date
    df["Date"] = pd.to_datetime(df["Kline Open Time"], unit="ms", errors="coerce")
    df = df.sort_values(by="Date", ascending=True)
    df.rename(
        columns={
            "Open Price": "Open",
            "High Price": "High",
            "Low Price": "Low",
            "Close Price": "Close",
        },
        inplace=True,
    )
    df.set_index("Date", inplace=True)
    df.drop(df.columns[[0, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True)

    return df


class BinanceDataLoader:
    def __init__(self, start_time: dt.datetime, end_time: dt.datetime):
        self.client = Client(os.environ.get("API_KEY"), os.environ.get("SECRET_KEY"))
        self.start_time = start_time
        self.end_time = end_time

    def get_data(self, asset_name):
        columns = [
            "Kline Open Time",
            "Open Price",
            "High Price",
            "Low Price",
            "Close Price",
            "Volume",
            "Kline Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
            "Unused",
        ]

        end_time = int(round(self.end_time.timestamp() * 1000, 0))
        start_time = int(round(self.start_time.timestamp() * 1000, 0))
        df = pd.DataFrame(
            self.client.get_historical_klines(
                asset_name,
                Client.KLINE_INTERVAL_15MINUTE,
                start_str=start_time,
                end_str=end_time,
                klines_type=enums.HistoricalKlinesType.SPOT,
            ),
            columns=columns,
        )

        raw_server_time = self.client.get_server_time()
        server_time = dt.datetime.fromtimestamp(raw_server_time["serverTime"] / 1000.0)
        print("server time:", server_time)

        return fix_df(df)

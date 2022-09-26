import numpy as np
import pandas as pd

from app.helpers.plotting import plot_PVSRA_candles


class PVSRAStrategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_pvsra(self):
        # the pipeline is:
        # calculate average vol -> calculate climax -> calculate PVSRA -> calculate EMAs
        self.data = calculate_average_vol(self.data)
        self.data = calculate_climax(self.data)
        self.data = calculate_pvsra(self.data)
        self.data = calculate_emas(self.data)

    def plot_data(self):
        plot_PVSRA_candles(self.data,
                           self.data[self.data['PVSRA'] == 'red'],
                           self.data[self.data['PVSRA'] == 'blue'],
                           self.data[self.data['PVSRA'] == 'green'],
                           self.data[self.data['PVSRA'] == 'pink'])


def calculate_average_vol(df):
    df["Av"] = df["Volume"].rolling(window=10).mean()

    return df


def calculate_climax(df):
    df["Value2"] = df["Volume"] * abs((df["High"] - df["Low"]))
    df["HiValue2"] = df["Value2"].rolling(window=10).max()
    df["Volume > 2 * Average"] = df["Volume"] >= (df["Av"] * 2)
    df["Spread > HiValue2"] = df["Value2"] >= df["HiValue2"]

    return df


def calculate_emas(df, emas=(5, 13, 50, 200)):
    # calculate emas
    for ema in emas:
        df[str(ema) + " Day EMA"] = (
            df["Close"].ewm(span=ema, adjust=False, ignore_na=False).mean()
        )

    return df


def calculate_pvsra(df: pd.DataFrame):
    df["isBull"] = (df["Close"] > df["Open"]).astype(int)
    df["isBull"] = df["isBull"].map({0: "no", 1: "yes"})

    climax = (df["Volume"] >= df["Av"] * 2) | (df["Value2"] >= df["HiValue2"])

    rising = df["Volume"] >= (df["Av"] * 1.5)

    condlist = [climax, rising]
    choicelist = [2, 1]

    # calculated each category
    df["VA"] = np.select(condlist=condlist, choicelist=choicelist, default=0)

    green = (df['isBull'] == 'yes') & (df['VA'] == 2)
    blue = (df['isBull'] == 'yes') & (df['VA'] == 1)

    red = (df['isBull'] != 'yes') & (df['VA'] == 2)
    pink = (df['isBull'] != 'yes') & (df['VA'] == 1)

    condlist = [green, blue, red, pink]
    choicelist = [0, 1, 2, 3]

    df['PVSRA'] = np.select(condlist, choicelist, default=-1)
    df['PVSRA'] = df['PVSRA'].map({
        -1: 'normal', 0: 'green', 1: 'blue', 2: 'red', 3: 'pink'
    })

    df.drop(columns=['VA', 'Value2', 'HiValue2'], inplace=True)

    return df

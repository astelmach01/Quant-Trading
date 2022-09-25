import datetime as dt
import os

import numpy as np
import pandas as pd

from binance import Client, enums
import plotly.graph_objects as go

API_KEY = os.environ.get('API_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')

client = Client(API_KEY, SECRET_KEY)

columns = ['Kline Open Time', 'Open Price', 'High Price', 'Low Price', 'Close Price',
           'Volume', 'Kline Close Time', 'Quote Asset Volume', 'Number of Trades',
           'Taker Buy Base Asset Volume',
           'Taker Buy Quote Asset Volume', 'Unused']

now = dt.datetime.now(dt.timezone.utc)
past = now - dt.timedelta(days=2)
past_timestamp_ms = int(round(past.timestamp() * 1000, 0))
df = pd.DataFrame(client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE,
                                               start_str=past_timestamp_ms,
                                               klines_type=enums.HistoricalKlinesType.SPOT),
                  columns=columns)

raw_server_time = client.get_server_time()
server_time = dt.datetime.fromtimestamp(raw_server_time['serverTime'] / 1000.0)
print("server time:", server_time)


def fix_df(df):
    # make them numbers
    df = df.apply(pd.to_numeric, errors='coerce')
    # get the date
    df['Date'] = pd.to_datetime(df['Kline Open Time'], unit='ms', errors='coerce')
    df = df.sort_values(by='Date', ascending=True)
    df.rename(columns={'Open Price': 'Open', 'High Price': 'High', 'Low Price': 'Low',
                       'Close Price': 'Close'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.drop(df.columns[[0, 6, 7, 8, 9, 10, 11]], axis=1, inplace=True)

    return df


df = fix_df(df)

if dt.datetime(year=2022, month=9, day=25, hour=22, minute=15) in df.index:
    x = df.loc['2022-09-25 22:15']['Volume']
    assert 4_700 < x < 4_800


def calculate_average_vol(df):
    df['Av'] = df['Volume'].rolling(window=10).mean()

    return df


df = calculate_average_vol(df)


def calculate_climax(df):
    df['Value2'] = df['Volume'] * abs((df['High'] - df['Low']))
    df['HiValue2'] = df['Value2'].rolling(window=10).max()
    df['Volume > 2 * Average'] = df['Volume'] >= (df['Av'] * 2)
    df['Spread > HiValue2'] = df['Value2'] >= df['HiValue2']

    return df


df = calculate_climax(df)


def calculate_emas(df, emas=(5, 13, 50, 200)):
    # calculate emas
    for ema in emas:
        df[str(ema) + ' Day EMA'] = df['Close'].ewm(span=ema, adjust=False, ignore_na=False).mean()

    return df


df = calculate_emas(df)


def calculate_pvsra(df: pd.DataFrame):
    df['isBull'] = (df['Close'] > df['Open']).astype(int)
    df['isBull'] = df['isBull'].map({0: 'no', 1: 'yes'})

    climax = (df['Volume'] >= df['Av'] * 2) | \
             (df['Value2'] >= df['HiValue2'])

    rising = df['Volume'] >= (df['Av'] * 1.5)

    condlist = [climax, rising]
    choicelist = [2, 1]

    df['VA'] = np.select(condlist=condlist, choicelist=choicelist, default=0)

    return df


df = calculate_pvsra(df)

# exponential_moving_averages = mpf.make_addplot(
#     df[['5 Day EMA', '13 Day EMA', '50 Day EMA', '200 Day EMA']])
# mpf.plot(df, type='candle', volume=True, show_nontrading=True, addplot=exponential_moving_averages)

bull = df[df['isBull'] == 'yes']
bear = df[df['isBull'] != 'yes']

df_green = bull[(bull['VA'] == 2)]
df_blue = bull[(bull['VA'] == 1)]

df_red = bear[(bear['VA'] == 2)]
df_pink = bear[(bear['VA'] == 1)]

print('Date Range: ', min(df.index), ':', max(df.index))
print('Number of red', len(df_red))
print('Number of blue', len(df_blue))
print('Number of green', len(df_green))
print('Number of pink', len(df_pink))


def create_candlestick_chart(dataframe: pd.DataFrame):
    return go.Candlestick(x=dataframe.index,
                          open=dataframe['Open'],
                          high=dataframe['High'],
                          low=dataframe['Low'],
                          close=dataframe['Close'])


fig = go.Figure(create_candlestick_chart(df))

fig.add_traces(create_candlestick_chart(df_red))

fig.add_traces(create_candlestick_chart(df_blue))

fig.add_traces(create_candlestick_chart(df_green))

fig.add_traces(create_candlestick_chart(df_pink))

# red
fig.data[1].increasing.fillcolor = 'rgb(255,0,0)'
fig.data[1].increasing.line.color = 'rgb(255,0,0)'
fig.data[1].decreasing.fillcolor = 'rgb(255,0,0)'
fig.data[1].decreasing.line.color = 'rgb(255,0,0)'

# blue
fig.data[2].increasing.fillcolor = 'rgba(14, 0, 172, 1)'
fig.data[2].increasing.line.color = 'rgba(14, 0, 172, 1)'
fig.data[2].decreasing.fillcolor = 'rgba(14, 0, 172, 1)'
fig.data[2].decreasing.line.color = 'rgba(14, 0, 172, 1)'

# green
fig.data[3].increasing.fillcolor = 'rgb(0, 255, 0)'
fig.data[3].increasing.line.color = 'rgb(0, 255, 0)'
fig.data[3].decreasing.fillcolor = 'rgb(0, 255, 0)'
fig.data[3].decreasing.line.color = 'rgb(0, 255, 0)'

# pink
fig.data[4].increasing.fillcolor = 'rgba(234, 0, 255, 1)'
fig.data[4].increasing.line.color = 'rgba(234, 0, 255, 1)'
fig.data[4].decreasing.fillcolor = 'rgba(234, 0, 255, 1)'
fig.data[4].decreasing.line.color = 'rgba(234, 0, 255, 1)'

fig.show()

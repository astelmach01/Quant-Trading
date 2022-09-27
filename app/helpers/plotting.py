import pandas as pd
import plotly.graph_objects as go


def create_candlestick_chart(dataframe: pd.DataFrame):
    return go.Candlestick(
        x=dataframe.index,
        open=dataframe["Open"],
        high=dataframe["High"],
        low=dataframe["Low"],
        close=dataframe["Close"],
    )


def plot_PVSRA_candles(
    df: pd.DataFrame,
    df_red: pd.DataFrame,
    df_blue: pd.DataFrame,
    df_green: pd.DataFrame,
    df_pink: pd.DataFrame,
):
    fig = go.Figure(create_candlestick_chart(df))

    fig.add_traces(create_candlestick_chart(df_red))

    fig.add_traces(create_candlestick_chart(df_blue))

    fig.add_traces(create_candlestick_chart(df_green))

    fig.add_traces(create_candlestick_chart(df_pink))

    def fill_with_color(fig_data, color: str):
        fig_data.increasing.fillcolor = color
        fig_data.increasing.line.color = color
        fig_data.decreasing.fillcolor = color
        fig_data.decreasing.line.color = color

    # red, blue, green, pink
    colors = (
        "rgb(255,0,0)",
        "rgba(14, 0, 172, 1)",
        "rgb(0, 255, 0)",
        "rgba(234, 0, 255, 1)",
    )
    for i in range(len(colors)):
        fill_with_color(fig.data[i + 1], colors[i])

    fig.show()

import datetime

import rich_click as click

from app.data_loader.binance_data import BinanceDataLoader
from app.strategies.pvsra import PVSRAStrategy


@click.group("app")
def cli():
    pass


@cli.command("run")
@click.option(
    "--start-date",
    type=click.DateTime(),
    help="the start date that which the algorithm runs in, required",
)
@click.option("--end-date",
              type=click.DateTime(),
              help="the end time inclusive, defaults to the current time if not set")
@click.option(
    "--asset-name",
    type=str,
    help="The asset to run on, like BTCUSDT",
)
def run(start_date: datetime.datetime, end_date: datetime.datetime, asset_name: str):
    if not start_date:
        raise ValueError('must provide a start date')

    if not end_date:
        end_date = datetime.datetime.utcnow()

    data = BinanceDataLoader(start_date, end_date).get_data(asset_name)

    strat = PVSRAStrategy(data)
    strat.calculate_pvsra()
    strat.plot_data()

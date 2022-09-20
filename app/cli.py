import datetime

import rich_click as click
from app.strategies.pvsra import PVSRAStrategy
from app.data_loader.binance_data import BinanceDataLoader


@click.group('app')
def cli():
    pass


@cli.command('run')
@click.option('--start-date',
              type=click.DateTime(),
              help='the start date that which the algorithm runs in')
@click.option('--end-date',
              type=click.DateTime(),
              help='the end time (exclusive')
def run(start_date: datetime.datetime, end_date: datetime.datetime):

    strat = PVSRAStrategy()

    loader = BinanceDataLoader()
    loader.get_data('ETHUSDT')

    strat.run()

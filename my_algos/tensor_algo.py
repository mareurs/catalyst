# Run Command
# catalyst run --start 2017-1-1 --end 2017-11-1 -o talib_simple.pickle \
#   -f talib_simple.py -x poloniex
#
# Description
# Simple TALib Example showing how to use various indicators
# in you strategy. Based loosly on
# https://github.com/mellertson/talib-macd-example/blob/master/talib-macd-matplotlib-example.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib as ta
from logbook import Logger
from matplotlib.dates import date2num
from matplotlib.finance import candlestick_ohlc

from catalyst import run_algorithm
from catalyst.api import (
    order,
    order_target_percent,
    symbol,
)
from catalyst.exchange.utils.stats_utils import get_pretty_stats
import tensorflow as tf
from tensor.tensor_model import TensorModel

algo_namespace = 'tensorflow'
log = Logger(algo_namespace)
filename = os.path.splitext(os.path.basename('tensor'))[0]
saveData = pd.DataFrame()

def initialize(context):
    log.info('Starting Tensorflow')

    context.ASSET_NAME = 'BTC_USD'
    context.asset = symbol(context.ASSET_NAME)

    context.ORDER_SIZE = 0.1
    context.SLIPPAGE_ALLOWED = 0.05

    context.swallow_errors = True
    context.errors = []

    # Bars to look at per iteration should be bigger than SMA_SLOW
    context.BARS = 60*24*10 # 10 days
    context.COUNT = 0

    # Technical Analysis Settings
    context.SMA_FAST = 50
    context.SMA_SLOW = 100
    context.RSI_PERIOD = 14
    context.RSI_OVER_BOUGHT = 80
    context.RSI_OVER_SOLD = 20
    context.RSI_AVG_PERIOD = 15
    context.MACD_FAST = 12
    context.MACD_SLOW = 26
    context.MACD_SIGNAL = 9
    context.STOCH_K = 14
    context.STOCH_D = 3
    context.STOCH_OVER_BOUGHT = 80
    context.STOCH_OVER_SOLD = 20

    context.tensorModel = TensorModel()
    context.tensorModel.initialize()
    context.history = [[]]
    context.histIndex = 0
    pass


def _handle_data(context, data):
    # Get price, open, high, low, close
    prices = data.history(
        context.asset,
        bar_count=context.BARS,
        fields=['price', 'open', 'high', 'low', 'close'],
        frequency='1m')

    # Create a analysis data frame
    analysis = pd.DataFrame(index=prices.index)

    # SMA FAST
    analysis['sma_f'] = ta.SMA(prices.close.as_matrix(), context.SMA_FAST)
    # SMA SLOW
    analysis['sma_s'] = ta.SMA(prices.close.as_matrix(), context.SMA_SLOW)

    # Relative Strength Index
    analysis['rsi'] = ta.RSI(prices.close.as_matrix(), context.RSI_PERIOD)
    # RSI SMA
    analysis['sma_r'] = ta.SMA(analysis.rsi.as_matrix(),
                               context.RSI_AVG_PERIOD)

    # MACD, MACD Signal, MACD Histogram
    analysis['macd'], analysis['macdSignal'], analysis['macdHist'] = ta.MACD(
        prices.close.as_matrix(), fastperiod=context.MACD_FAST,
        slowperiod=context.MACD_SLOW, signalperiod=context.MACD_SIGNAL)

    # Stochastics %K %D
    # %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    # %D = 3-day SMA of %K
    analysis['stoch_k'], analysis['stoch_d'] = ta.STOCH(
        prices.high.as_matrix(), prices.low.as_matrix(),
        prices.close.as_matrix(), slowk_period=context.STOCH_K,
        slowd_period=context.STOCH_D)

    # SMA FAST over SLOW Crossover
    analysis['sma_test'] = np.where(analysis.sma_f > analysis.sma_s, 1, 0)

    # MACD over Signal Crossover
    analysis['macd_test'] = np.where((analysis.macd > analysis.macdSignal), 1,
                                     0)

    # Stochastics OVER BOUGHT & Decreasing
    analysis['stoch_over_bought'] = np.where(
        (analysis.stoch_k > context.STOCH_OVER_BOUGHT) & (
            analysis.stoch_k > analysis.stoch_k.shift(1)), 1, 0)

    # Stochastics OVER SOLD & Increasing
    analysis['stoch_over_sold'] = np.where(
        (analysis.stoch_k < context.STOCH_OVER_SOLD) & (
            analysis.stoch_k > analysis.stoch_k.shift(1)), 1, 0)

    # RSI OVER BOUGHT & Decreasing
    analysis['rsi_over_bought'] = np.where(
        (analysis.rsi > context.RSI_OVER_BOUGHT) & (
            analysis.rsi < analysis.rsi.shift(1)), 1, 0)

    # RSI OVER SOLD & Increasing
    analysis['rsi_over_sold'] = np.where(
        (analysis.rsi < context.RSI_OVER_SOLD) & (
            analysis.rsi > analysis.rsi.shift(1)), 1, 0)

    # Save the prices and analysis to send to analyze
    context.prices = prices
    context.analysis = analysis
    context.price = data.current(context.asset, 'price')

    # Log the values of this bar
    #logAnalysis(analysis)

    analysis['price'] = prices.price
    analysis['high'] = prices.high
    analysis['low'] = prices.low
    # analysis['next_10_high'] = -1
    # analysis['next_10_low'] = -1
    # analysis['next_50_high'] = -1
    # analysis['next_50_low'] = -1
    # analysis['next_100_high'] = -1
    # analysis['next_100_low'] = -1

#    populate_expected_values(analysis)
#    analysis.to_csv(filename + '.csv')

    print "Current price: " + str(context.price)

    result = context.tensorModel.compute(analysis.loc[analysis.index.values[-1]])
    context.history.append(result)
    context.histIndex = context.histIndex+1
    if context.histIndex > 10:
        print("Real: ", getLast(analysis, 'high'), getLast(analysis, 'low'))
        print("Computed: ", context.history[context.histIndex - 10])

    makeOrders(context, analysis)


def populate_expected_values(analysis):
    for i in range(analysis.index.size - 100):

        now = analysis.index.values[i]
        next10 = analysis.index.values[i + 10];
        next50 = analysis.index.values[i + 50];
        next100 = analysis.index.values[i + 100];

        analysis.loc[now,'next_10_low'] = analysis.loc[next10, 'low']
        analysis.loc[now,'next_10_high'] = analysis.loc[next10, 'high']
        analysis.loc[now,'next_50_low'] = analysis.loc[next50, 'low']
        analysis.loc[now,'next_50_high'] = analysis.loc[next50, 'high']
        analysis.loc[now,'next_100_low'] = analysis.loc[next100, 'low']
        analysis.loc[now,'next_100_high'] = analysis.loc[next100, 'high']



def handle_data(context, data):
    #log.info('handling bar {}'.format(data.current_dt))
    try:
        _handle_data(context, data)
    except Exception as e:
        log.warn('aborting the bar on error {}'.format(e))
        context.errors.append(e)

    log.info('completed bar {}, total execution errors {}'.format(
        data.current_dt,
        len(context.errors)
    ))

    if len(context.errors) > 0:
        log.info('the errors:\n{}'.format(context.errors))


def analyze(context, results):
    filename = os.path.splitext(os.path.basename('results_tensor'))[0]
    results.to_csv(filename + '.csv')

    # Save results in CSV file
    #filename = os.path.splitext(os.path.basename('tensor_simple'))[0]
    #results.to_csv(filename + '.csv')

    log.info('the daily stats:\n{}'.format(get_pretty_stats(results)))
    chart(context, context.prices, context.analysis, results)
    pass


def makeOrders(context, analysis):
    if context.asset in context.portfolio.positions:

        # Current position
        position = context.portfolio.positions[context.asset]

        if (position == 0):
            log.info('Position Zero')
            return

        # Cost Basis
        cost_basis = position.cost_basis

        log.info(
            'Holdings: {amount} @ {cost_basis}'.format(
                amount=position.amount,
                cost_basis=cost_basis
            )
        )

        # Sell when holding and got sell singnal
        if isSell(context, analysis):
            profit = (context.price * position.amount) - (
                cost_basis * position.amount)
            order_target_percent(
                asset=context.asset,
                target=0,
                limit_price=context.price * (1 - context.SLIPPAGE_ALLOWED),
            )
            log.info(
                'Sold {amount} @ {price} Profit: {profit}'.format(
                    amount=position.amount,
                    price=context.price,
                    profit=profit
                )
            )
        else:
            log.info('no buy or sell opportunity found')
    else:
        # Buy when not holding and got buy signal
        if isBuy(context, analysis):
            order(
                asset=context.asset,
                amount=context.ORDER_SIZE,
                limit_price=context.price * (1 + context.SLIPPAGE_ALLOWED)
            )
            log.info(
                'Bought {amount} @ {price}'.format(
                    amount=context.ORDER_SIZE,
                    price=context.price
                )
            )

def isBuy(context, analysis):
    if context.price < context.history[-1][1]:
        return True

    return False

def isSell(context, analysis):
    if context.price > context.history[-1][0]:
        return True

    return False


def chart(context, prices, analysis, results):
    results.portfolio_value.plot()

    # Data for matplotlib finance plot
    dates = date2num(prices.index.to_pydatetime())

    # Create the Open High Low Close Tuple
    prices_ohlc = [tuple([dates[i],
                          prices.open[i],
                          prices.high[i],
                          prices.low[i],
                          prices.close[i]]) for i in range(len(dates))]

    fig = plt.figure(figsize=(14, 18))

    # Draw the candle sticks
    ax1 = fig.add_subplot(411)
    ax1.set_ylabel(context.ASSET_NAME, size=20)
    candlestick_ohlc(ax1, prices_ohlc, width=0.4, colorup='g', colordown='r')

    # Draw Moving Averages
    analysis.sma_f.plot(ax=ax1, c='r')
    analysis.sma_s.plot(ax=ax1, c='g')

    # RSI
    ax2 = fig.add_subplot(412)
    ax2.set_ylabel('RSI', size=12)
    analysis.rsi.plot(ax=ax2, c='g',
                      label='Period: ' + str(context.RSI_PERIOD))
    analysis.sma_r.plot(ax=ax2, c='r',
                        label='MA: ' + str(context.RSI_AVG_PERIOD))
    ax2.axhline(y=30, c='b')
    ax2.axhline(y=50, c='black')
    ax2.axhline(y=70, c='b')
    ax2.set_ylim([0, 100])
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    # Draw MACD computed with Talib
    ax3 = fig.add_subplot(413)
    ax3.set_ylabel('MACD: ' + str(context.MACD_FAST) + ', ' + str(
        context.MACD_SLOW) + ', ' + str(context.MACD_SIGNAL), size=12)
    analysis.macd.plot(ax=ax3, color='b', label='Macd')
    analysis.macdSignal.plot(ax=ax3, color='g', label='Signal')
    analysis.macdHist.plot(ax=ax3, color='r', label='Hist')
    ax3.axhline(0, lw=2, color='0')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)

    # Stochastic plot
    ax4 = fig.add_subplot(414)
    ax4.set_ylabel('Stoch (k,d)', size=12)
    analysis.stoch_k.plot(ax=ax4, label='stoch_k:' + str(context.STOCH_K),
                          color='r')
    analysis.stoch_d.plot(ax=ax4, label='stoch_d:' + str(context.STOCH_D),
                          color='g')
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels)
    ax4.axhline(y=20, c='b')
    ax4.axhline(y=50, c='black')
    ax4.axhline(y=80, c='b')

    plt.show()


def logAnalysis(analysis):
    return

def getLast(arr, name):
    return arr[name][arr[name].index[-1]]


if __name__ == '__main__':
    _start = pd.to_datetime('2018-01-20', utc=True)
    _end = pd.to_datetime('2018-01-20', utc=True)

    run_algorithm(
        capital_base=10000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='bitfinex',
        base_currency='usd',
        start=_start,
        end=_end,
    )

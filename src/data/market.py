import os
import datetime
import numpy as np
import pandas as pd
import pytz
from tqdm.auto import tqdm

from .base import Midprice


PATH = "/scratch/perez/double_liquidation"


def _compute_dt(seconds, date):
    tz = pytz.timezone("US/Eastern")
    d = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(0, seconds)
    d_tz = d.replace(tzinfo=tz)
    return d_tz.timestamp() - d.timestamp()

def load_equity(date, symbol="AAPL"):
    #data = pd.read_csv(os.path.join(PATH, symbol,
    #                  "{}_{}_34200000_57600000_orderbook_10.csv".format(symbol, date)),
    #                   header=None, usecols=[0, 2])
    
    data = pd.read_hdf(os.path.join(PATH, symbol,
                      "{}_{}_34200000_57600000_orderbook_10.h5".format(symbol, date)),
                       header=None, usecols=[0, 2])
    data = data.iloc[:, [0, 2]]
    data.columns = ["bid", "ask"]

    #messages = pd.read_csv(os.path.join(PATH, symbol,
    #                       "{}_{}_34200000_57600000_message_10.csv".format(symbol, date)),
    #                        header=None, usecols=[0])
    messages = pd.read_hdf(os.path.join(PATH, symbol,
                           "{}_{}_34200000_57600000_message_10.h5".format(symbol, date)),
                            header=None, usecols=[0])
    messages = messages.iloc[:, [0]]

    #data["equity"] = 1e-4 * 0.5 * (data["bid"] + data["ask"])
    data[symbol] = 1e-4 * data["ask"]

    #data["ts"] = messages[0]
    data["ts"] = messages
    
    data = data[["ts", symbol]]
    data.drop_duplicates("ts", keep="last", inplace=True)

    dt = _compute_dt(data["ts"].values[0], date)
    date_ts = datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()
    data["ts"] += dt + date_ts
    data.drop_duplicates("ts", keep="last", inplace=True)

    data.set_index("ts", inplace=True)

    return data

def load_fx(date, symbol="GBPUSD"):
    #data = pd.read_csv(os.path.join(PATH, symbol, "Ldn_{}_TOB_{}.csv".format(date, symbol)))
    data = pd.read_hdf(os.path.join(PATH, symbol, "Ldn_{}_TOB_{}.h5".format(date, symbol)))
    
    data = data[["timestamp_micros", "bid_price", "ask_price"]]
    data.columns = ["ts"] + data.columns[1:].tolist()
    data[symbol] = 1e-6 * 0.5 * (data["ask_price"] + data["bid_price"])
    data = data[["ts", symbol]]
    data["ts"] /= 1e6
    data.drop_duplicates("ts", keep="last", inplace=True)

    data.set_index("ts", inplace=True)

    return data

def load(date, equity=("AAPL",), fx=("GBPUSD",)):
    equity_data = [load_equity(date, symbol=s) for s in equity]
    fx_data = [load_fx(date, symbol=s) for s in fx]

    data = pd.concat(equity_data + fx_data, axis=1, sort=False)
    data = data.fillna(method="ffill").dropna()

    return data

class Market(Midprice):
    """Market data."""

    def __init__(self, equity="AAPL", fx="GBPUSD", invert=False, invert_main=False):
        self.equity = equity
        self.fx = fx
        self.invert = invert
        self.invert_main = invert_main

    def _generate_paths(self, start, end, time_start, window_length, order=6, **kwargs):
        start = datetime.datetime.strptime(start, "%Y-%m-%d")
        end = datetime.datetime.strptime(end, "%Y-%m-%d")

        paths = []

        n_days = (end - start).days
        pbar = tqdm(range(n_days))

        while start <= end:
            pbar.update()
            day = start.strftime("%Y-%m-%d")

            try:
                data = load(day, equity=self.equity, fx=self.fx)
            except FileNotFoundError:
                start += datetime.timedelta(1)
                continue


            window_start = datetime.datetime.strptime(day + time_start, "%Y-%m-%d%H:%M")
            window_end = window_start + datetime.timedelta(0, 60 * window_length)

            try:
                data = data.loc[window_start.timestamp() : window_end.timestamp()]
            except:
                start += datetime.timedelta(1)
                continue

            if len(data) == 0:
                start += datetime.timedelta(1)
                continue

            timeline = data.index - data.index[0]
            timeline /= timeline[-1]

            path = np.c_[timeline, data.values]

            # Permute last column with 3rd column.
            # The path will have the following dimensions:
            # Time, Main asset, bridging asset, extra info 1, ..., extra info n.
            path[:, 2], path[:, -1] = path[:, -1].copy(), path[:, 2].copy()

            if np.min(path) >= 0:
                # If this isn't satisfied something funny happened
                for i in range(1, path.shape[1]):
                    path[:, i] /= path[0, i]

                if self.invert:
                    # FX is in B/A format -- convert to A/B.
                    path[:, 2] = 1. / path[:, 2]
                if self.invert_main:
                    # Idem.
                    path[:, 1] = 1. / path[:, 1]


                paths.append(path)

            start += datetime.timedelta(1)

        return paths

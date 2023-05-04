
import yfinance as yf
import json
import datetime
import pandas as pd
import multiprocessing
import os
from time import sleep

def changer1(now):
    return int(now.value/1e6)
def changer2(val):
    return int(val)
def changer3(val):
    return val.strftime('%Y-%m-%d')
def changer4(val):
    temp = []
    for time,value in val:
        temp.append({"time":time,"value":value})
    return temp
def get_price(name: str):
    if name.isnumeric(): return "Invalid Stock Name"
    try:
        ticker = yf.Ticker(name)
        history = ticker.history("1mo")
        more_info = ticker.info
        t1 = list(history.index)
        t2 = list(history.Close)
        t3 = list(map(changer1,t1))
        t4 = list(map(changer3,t1))
        t5 = list(map(changer2, t2))
        t6 = list(zip(t3,t5))
        t7 = changer4(list(zip(t4,t5)))
        res = {"price_data": t6,
            "price_data_2": t7,
            "more_info": more_info
            }
        path = name.split(".")[0]
        with open("data/yfinance_stock/{}.json".format(path), 'w') as f:
            json.dump(res, f, indent=4)
    except:
        print("Data Not Found")

def get_prices():
    symbols = pd.read_csv("data/stock_info.csv")["Symbol"].tolist()
    symbols = [symbol + ".NS" for symbol in symbols]
    print("Done 1")
    pool = multiprocessing.Pool(os.cpu_count())
    pool.map(get_price, symbols)
    pool.close()





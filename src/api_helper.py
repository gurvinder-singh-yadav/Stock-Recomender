from datetime import date
import shutil
import os
import pandas as pd
import glob
import datetime
import requests
from bs4 import BeautifulSoup
import re
import json
import numpy as np

def remove_percent_symbol(x):
    return x.rstrip("%")

def concat_grow_funds():
    today = str(date.today())
    today_dir = os.path.join("data","grow" ,today)
    funds_dir = os.path.join(today_dir, "funds")
    if os.path.exists(funds_dir):
        files = glob.glob(os.path.join(funds_dir, "*.csv"))
        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file, index_col=None))
        shutil.rmtree(funds_dir)
        funds = pd.concat(dfs, ignore_index=True)
        funds.to_csv(os.path.join(today_dir, "funds.csv"), index = None)

def concat_grow_stocks():
    today = str(date.today())
    today_dir = os.path.join("data","grow" ,today)
    stocks_dir = os.path.join(today_dir, "stocks")
    if os.path.exists(stocks_dir):
        files = glob.glob(os.path.join(stocks_dir, "*.csv"))
        dfs = []
        for file in files:
            dfs.append(pd.read_csv(file, index_col=None))
        shutil.rmtree(stocks_dir)
        stocks = pd.concat(dfs, ignore_index=True)
        stocks = stocks.drop_duplicates(subset=["name", "href"])
        stocks.to_csv(os.path.join(today_dir, "stocks_index.csv"), index = None)

def get_news_data(date):
    df = pd.read_csv("data/news_processed.csv")
    # return [date, df["date"].iloc[0]]
    df = df[df["date"] == date]
    print(df.head())

    # return df
    return df.to_dict("list") 

def convert_date(date: str):
    # print(date)
    format = "%b %d, %Y, %I:%M %p %Z"
    date = datetime.datetime.strptime(date, format)
    return date.strftime("%Y-%m-%d")   

def process_news():
    df = pd.read_csv("data/news.csv")
    print("Processing...")
    df = df.dropna()
    df =  df[df["title"].map(lambda x: not(any(map(str.isnumeric, x.split(" "))))) == True]
    df["fund_name"] = df["title"].apply(lambda x: x.split(":")[-1].strip())
    df["action"] = df["title"].apply(lambda x: x.split(",")[0].split(" ")[0] if "," in x else " ")
    df["target"] = df["title"].apply(lambda x: x.split(":")[-2].split(" ")[-1] if(":" in x) else " ")
    df["stock_name"] = df["title"].apply(lambda x: " ".join(x.split(",")[0].split(" ")[1:]) if "," in x else " ")
    df["date"] = df["time"].apply(convert_date)
    del df["time"]
    df = df.fillna("No Info")
    df.to_csv("data/news_processed.csv", index=None)

def remove_missing_stocks():
    def have_data(name):
        return name in names
    info_df = pd.read_csv("data/stock_info.csv")
    names = info_df["name"].tolist()
    del info_df
    today = str(datetime.date.today())
    df = pd.read_csv("data/grow/{}/funds.csv".format(today))
    df = df[df["Name"].map(have_data)]
    df.to_csv("data/grow/{}/funds.csv".format(today), index=None)


def get_market_news_data():
    URL = "https://www.moneycontrol.com/news/stocksinnews-142.html"
    r = requests.get(URL) 

    soup = BeautifulSoup(r.content, 'html5lib') 

    table = soup.findAll('li', attrs = {'class':'clearfix'}) 
    temp = []
    for row in table:
        link = row.find('a')['href']
        title =  row.find('a')['title']
        text = row.find('p').text
        temp.append({"link":link,"title":title,"text":text})
    return temp

def concat_yf_stock_info():
    files = glob.glob("data/yfinance_stock/*")
    stock = {}
    # print(files)
    for file in files:
        with open(file, "r") as f:
            # print(file)
            js = json.load(f)
            # print(js)
        # print(js)
        stock[file.split(".")[0].split("/")[-1]] = js
    with open("data/yf_stock_info.json", 'w') as f:
        json.dump(stock, f)
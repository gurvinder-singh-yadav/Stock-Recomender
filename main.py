from src.grow_scraper import *
from src.ticker_scraper import get_tick_funds, get_tick_links
from src.economic_times import get_stories
from src.api_helper import *
from src.yfinance_scraper import get_prices
import os
import pandas as pd
import datetime
def concat(dir):
    today = str(datetime.date.today())
    index_path = "data/grow/{}/index.csv".format(today)
    if os.path.exists(index_path):
        return "Done"
    paths = os.listdir(dir)
    dfs = []
    # print(paths)
    if len(paths) > 1:
        for path in paths:
            path = os.path.join(dir, path)
            dfs.append(pd.read_csv(path, index_col=None))
            os.remove(path)
        df = pd.concat(dfs)
        os.rmdir(dir)
        df.to_csv(index_path, index=None)
    return "Done"


if __name__ == "__main__":
    save_summaries()
    today = str(datetime.date.today())
    concat("data/grow/{}/index".format(today))
    scrape_grow_funds()
    concat_grow_funds()
    concat_grow_stocks()
    get_stocks_info()
    summarise_stocks()
    remove_missing_stocks()
    get_tick_links()
    get_tick_funds()
    get_stories(50)
    process_news()
    get_prices()
    concat_yf_stock_info()
    
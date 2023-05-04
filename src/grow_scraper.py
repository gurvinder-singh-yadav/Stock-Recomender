from selenium import webdriver
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from bs4 import BeautifulSoup
import datetime
import time
import glob
import shutil
import os
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
import json
from time import sleep


def chunker(n):
    k = n // os.cpu_count()
    for i in range(0,n, k):
        yield (i, i + k)

def get_page_summary(start, stop):
    """
    scrape mutual fund indices of pages from range(start, stop)
    """
    driver = webdriver.Chrome(
    ChromeDriverManager().install(),
    )
    for page in tqdm(range(start, stop)):
        base_url = "https://groww.in/mutual-funds/filter?q=&fundSize=&category=%5B%22Equity%22%5D&pageNo={}&sortBy=3".format(page)
        driver.get(base_url)
        try:
            table = driver.find_element(By.CLASS_NAME, "tb10Table")
            elements = table.find_elements(By.TAG_NAME, "tr")
            data = {
                "name":[],
                "link":[],
                "risk":[]
            }
            for element in elements:
                link = element.find_element(By.TAG_NAME, "a").get_attribute("href")
                td = element.find_elements(By.TAG_NAME, "td")
                fund = td[1].text.split("\n")
                data['link'].append(link)
                data["name"].append(fund[0])
                data["risk"].append(fund[1])
            date = str(datetime.date.today())
            root_dir = "data/grow"
            path = os.path.join(root_dir, date, "index", "{}.csv".format(page))
            data = pd.DataFrame(data)
            data.to_csv(path, index=None)
        except:
            continue
    driver.close()

def save_summaries(pages = 28):
    date = str(datetime.date.today())
    root_dir = "data/grow"
    index_path = os.path.join(root_dir, date, "index.csv")
    if os.path.exists(index_path):
        return "no need to scrape"
    if not os.path.exists(os.path.join(root_dir, date, "index")):
        os.makedirs(os.path.join(root_dir, date, "index"))
        pool = ThreadPool(os.cpu_count())
        pool.starmap(get_page_summary, chunker(28))
    return "scraped"

def concat(dir):
    paths = glob.glob(os.path.join(dir, "*.csv"))
    dfs = []
    today = str(datetime.date.today())
    # print(paths)
    if len(paths) > 1:
        for path in paths:
            path = os.path.join(dir, path)
            dfs.append(pd.read_csv(path, index_col=None))
            os.remove(path)
        df = pd.concat(dfs)
        df.to_csv(os.path.join("data/grow/{}".format(today), "funds_index.csv"), index=None)


def scrape_fund(start, stop):
    today = str(datetime.date.today())
    index_path = os.path.join("data/grow", today, "index.csv")
    index = pd.read_csv(index_path)
    names = index["name"].iloc[start: stop]
    urls = index["link"].iloc[start: stop]
    driver = webdriver.Chrome(
    ChromeDriverManager().install(),
        )
    stocks = {"name":[],
              "href":[]
              }
    for idx, (url, name) in tqdm(enumerate(zip(urls, names))):
        driver.get(url)
        try:
            table = driver.find_element(By.XPATH, "//div[@class='col l5 offset-l2 fnd2TableDiv']//table[@class='tb10Table fd12Table']")
            l1 = pd.read_html(table.get_attribute("outerHTML"))[0]
            l1 = l1.transpose()
            l1.columns = l1.iloc[0]
            l1 = l1.drop(l1.index[0])
            fund_size = float((l1["Fund size"].iloc[0].lstrip("₹").rstrip("Cr")).replace(",", ""))
            see_all = driver.find_element(By.XPATH, "//div[@class='holdings101Cta cur-po']")
            see_all.click()
            source = driver.page_source
            soup = BeautifulSoup(source , "lxml")
            table = soup.find("table", {"class":"tb10Table holdings101Table"})
            table_str = str(table)
            df = pd.read_html(table_str)[0]
            df["Assets(Rs_Cr.)"] = df["Assets"].apply(lambda x: float(x.rstrip("%")) * fund_size)
            del df["Assets"]
            df["funds_name"] = pd.Series([name]*df.values.shape[0])
            df.to_csv("data/grow/{}/funds/{}.csv".format(today, name),index =None)
            a_tags = table.find_all("a")
            stocks["name"].extend([a.text for a in a_tags])
            stocks["href"].extend(["https://groww.in" +a.get("href") for a in a_tags])
            stock_df = pd.DataFrame(stocks)
            stock_df.to_csv("data/grow/{}/stocks/{}.csv".format(today, start + idx), index=None)
        except:
            continue
    driver.close()

def scrape_grow_funds():
    today = str(datetime.date.today())
    if os.path.exists(os.path.join("data/grow",today, "funds.csv")):
        return "Data Upto Date"
    funds_dir = os.path.join("data/grow",today, "funds")
    stock_index_dir = os.path.join("data/grow",today, "stocks")
    if os.path.exists("data/grow/{}/stocks_index.csv") and os.path.exists("data/grow/{}/funds_index.csv"):
        return "already up to date"
    else:
        total_funds = len(pd.read_csv("data/grow/{}/index.csv".format(today)))
        if not os.path.exists(funds_dir): os.mkdir(funds_dir) 
        if not os.path.exists(stock_index_dir): os.mkdir(stock_index_dir)
        pool = ThreadPool(os.cpu_count())
        pool.starmap(scrape_fund, chunker(total_funds))
        return "Updated"

def get_stock_url(url):
    driver = webdriver.Chrome(
    ChromeDriverManager().install(),
        )
    driver.get(url)
    element = driver.find_element(By.XPATH, "//div[@class='holdings101Cta cur-po']")
    element.click()
    try: 
        table = driver.find_element(By.XPATH, "//table[@class='tb10Table holdings101Table']")
        trs = table.find_elements(By.TAG_NAME, 'tr')
        stock_names = []
        stock_url = []
        for i in range(1, len(trs)):
            try:
                name = trs[i].find_element(By.TAG_NAME, "td").text
                link = trs[i].find_element(By.TAG_NAME, "a").get_attribute("href")
                stock_names.append(name)
                stock_url.append(link)
            except:
                continue
        driver.close()
        df = pd.DataFrame({"Name":stock_names,
                        "Link": stock_url
                        })
        return df
    except:
        return None


def get_stock_details(name) :
    root_dir = "data/stocks"
    driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    )
    driver.get("https://groww.in/#")
    stock_details=[]
    driver.find_element(By.XPATH,"//input[@id='globalSearch23']").send_keys(name) 
    time.sleep(1)
    driver.find_element(By.XPATH,"//input[@id='globalSearch23']").send_keys(Keys.ENTER)
    time.sleep(5)
    table = driver.find_element(By.XPATH,"(//tbody)[1]")
    trs = table.find_elements(By.TAG_NAME,"tr")
    for tr in trs:
        td = tr.find_elements(By.TAG_NAME,"td")
        for tds in td:
         stock_details.append(tds.text)
    print(stock_details)
    col = []
    results = []
    col.append('Parameter')
    results.append('results')
    for i, res in enumerate(stock_details):
        if(i%2==0):
            col.append(res)
        else:
            results.append(res)
    dict1 = dict(zip(col, results))
    path = root_dir + "/{}.json".format(name)
    
    with open(path, "w") as f:
        json.dump(dict1, f, indent=4)
    driver.close()
    return 
    

def get_stock_url(url):
    driver = webdriver.Chrome(
    ChromeDriverManager().install(),
        )
    driver.get(url)
    today = str(datetime.date.today())
    element = driver.find_elements(By.XPATH, "//div[@class='holdings101Cta cur-po']")
    if len(element) < 1:
        driver.close()
        return None
    element[0].click()
    table = driver.find_element(By.XPATH, "//table[@class='tb10Table holdings101Table']")
    trs = table.find_elements(By.TAG_NAME, 'tr')
    stock_names = []
    stock_url = []
    for i in range(1, len(trs)):
        try:
            name = trs[i].find_element(By.TAG_NAME, "td").text
            link = trs[i].find_element(By.TAG_NAME, "a").get_attribute("href")
            stock_names.append(name)
            stock_url.append(link)
        except:
            continue
    driver.close()
    df = pd.DataFrame({"Name":stock_names,
                       "Link": stock_url
                       })
    if not os.path.exists("data/stocks"):
        os.mkdir("data/stocks")
    df.to_csv("data/stocks/{}.csv".format(url.split("/")[-1]), index=None)


def get_stock_urls():
    if os.path.exists("data/stocks.csv"):
        return "Already Upto Date"
    today = str(datetime.date.today())
    root_dir = "data/grow"
    path = os.path.join(root_dir, today, "index.csv")
    index = pd.read_csv(path)
    # dfs = map(get_stock_url, index["link"])
    pool = ThreadPool(os.cpu_count()*2)
    results = pool.map(get_stock_url,index["link"])

def concat_stock_urls():
    files = glob.glob("data/stock/*.csv")
    dfs = []
    print(files)
    for file in files:
        dfs.append(pd.read_csv(file, index_col=None))
    print(len(dfs))
    df = pd.concat(dfs, ignore_index=True)
    df = df[["Name", "Link"]]
    df.to_csv("data/stocks.csv", index=None)
    shutil.rmtree("data/stocks/")
    
def get_stock_info(start, stop):
        
        driver = webdriver.Chrome(
        ChromeDriverManager().install(),
                )
        today = str(datetime.date.today())

        df = pd.read_csv("data/grow/{}/stocks_index.csv".format(today))
        urls = df["href"].iloc[start:stop].tolist()
        names = df["name"].iloc[start:stop].tolist()
        stock = {}
        for idx, (name, url) in enumerate(zip(names, urls)):
                root_dir = "data/grow/{}/stock".format(today)
                path = os.path.join(root_dir, "{}.json".format(idx + start))
                driver.get(url)
                try:
                        element = driver.find_element(By.XPATH, "//body[1]/div[1]/div[2]/div[2]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[1]/div[1]/div[4]/div[1]/section[1]/section[1]/div[1]/div[1]/div[2]/div[1]/div[2]/div[1]")
                        stock["todaymin"] = element.get_attribute("aria-valuemin")
                        stock["todaymax"] =  element.get_attribute("aria-valuemax")
                        stock["todaynow"] = element.get_attribute("aria-valuenow")
                        element = driver.find_element(By.XPATH, "//body[1]/div[1]/div[2]/div[2]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[1]/div[1]/div[4]/div[1]/section[1]/section[1]/div[1]/div[1]/div[2]/div[2]/div[2]/div[1]")
                        stock["52min"] = element.get_attribute("aria-valuemin")
                        stock["52max"] =  element.get_attribute("aria-valuemax")
                        stock["52now"] = element.get_attribute("aria-valuenow")
                        xpath = "//body/div[@id='__next']/div[@id='root']/div/div[contains(@class,'container web-align')]/div[@class='row pw14Container']/div[@class='pw14MainWrapper']/div[@class='pw14ContentWrapper']/div[@class='col l12']/div[@class='container web-align']/div[@class='col l12 stkP12BelowFoldDiv']/div[@class='col l12 stkP12BelowFoldInnerCont']/div[@class='col l12']/div[@class='col l12']/div[@class='col l12']/section/section/div[@class='onMount-appear-done onMount-enter-done']/div[1]"
                        change_price = driver.find_element(By.XPATH, "//div[@class='lpu38Day fs14 primaryClr']")
                        stock["price_change"] = change_price.text
                        div = driver.find_element(By.XPATH, xpath)
                        innerhtml = div.get_attribute("innerHTML")
                        soup = BeautifulSoup(innerhtml, "html.parser")
                        names = [i.text for i in soup.find_all("div", {"class" : "stpp34KeyText stpp34KeyTextStk fs14"})]
                        values = [i.text for i in soup.find_all("span", {"class": "stpp34Value fs16"})]
                        element = driver.find_element(By.XPATH, "//table[@class='tb10Table col l12 ft785Table']")
                        trs = element.find_elements(By.TAG_NAME, "tr")
                        for i in range(len(trs)):
                                tds = trs[i].find_elements(By.TAG_NAME, "td")
                                names.append(tds[0].text)
                                values.append(tds[1].text)
                        xpath =  "//div[@class='col l12 stkP12Section onMount-appear-done onMount-enter-done']//table[1]"
                        element = driver.find_element(By.XPATH, xpath)
                        trs = element.find_elements(By.TAG_NAME, "tr")
                        for i in range(len(trs)):
                                tds = trs[i].find_elements(By.TAG_NAME, "td")
                                names.append(tds[0].text)
                                values.append(tds[1].text)
                        xpath = "//div[@class='col l12 stkP12Section onMount-appear-done onMount-enter-done']//table[2]"
                        element = driver.find_element(By.XPATH, xpath)
                        trs = element.find_elements(By.TAG_NAME, "tr")
                        for i in range(len(trs)):
                                tds = trs[i].find_elements(By.TAG_NAME, "td")
                                if "Symbol" in tds[0].text:
                                        names.append("Symbol")
                                else:
                                        names.append(tds[0].text)
                                values.append(tds[1].text)
                        stock.update(dict(zip(names, values)))
                        
                        
                        stock["name"] = name
                        # print(path)
                        print(path)
                        with open(path, 'w') as f:
                                json.dump(stock, f, indent=4)
                except:
                        continue
        driver.close()
        
def get_stocks_info():
    today = str(datetime.date.today())
    df = pd.read_csv("data/grow/{}/stocks_index.csv".format(today))
    root_dir = "data/grow/{}/stock".format(today)
    num_stocks = len(df)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if len(os.listdir(root_dir)) > 1:
        return "Already updated"
    pool = ThreadPool(os.cpu_count())
    pool.starmap(get_stock_info, chunker(num_stocks))

         


def summarise_stocks():
    today = str(datetime.date.today())
    files = glob.glob("data/grow/{}/stock/*.json".format(today))
    data = {}
    with open(files[0], "r") as f:
        js = json.load(f)
        keys = js.keys()
        for key in keys:
            data[key] = []
    for file in files:
            with open(file, "r") as f:
                js = json.load(f)
                for key in js.keys():
                        data[key].append(js.get(key, None))
    # for k, v in data.items():
    #     print(k, len(v))
    df = pd.DataFrame(data)
    df.to_csv("data/stock_info.csv", index=None)
        



    
    

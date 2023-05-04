from selenium import webdriver
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import datetime
import time
import os
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def get_tick_links():
    date = str(datetime.date.today())
    root_dir = "data/tick"
    index_path = os.path.join(root_dir, date, "index", "index.csv")
    if not os.path.exists(os.path.join(root_dir, date, "index")):
        os.makedirs(os.path.join(root_dir, date, "index"))
        driver = webdriver.Chrome(
        ChromeDriverManager().install(),
        )
        driver.get("https://ticker.finology.in/investor")
        fundsUrl = []
        table = driver.find_element(By.XPATH,"/html/body/form/div[5]/div[2]")
        aTags = table.find_elements(By.TAG_NAME, "a")
        for a in aTags:
                    link = a.get_attribute('href')
                    fundsUrl.append(link)    
        dict1 = {"url":(fundsUrl)}
        df = pd.DataFrame(dict1)
        df.to_csv(index_path, index=None)
        driver.close()

def get_tick_funds():
    root_dir = "data/tick"
    date = str(datetime.date.today())
    funds_path=os.path.join(root_dir, date, "funds")
    index_path = os.path.join(root_dir, date,"index", "index.csv" )
    if not os.path.exists(funds_path):
        os.makedirs(os.path.join(root_dir, date, "funds"))

        df = pd.read_csv(index_path)
        fundsUrl = df.values.reshape(-1)
        driver = webdriver.Chrome(
        ChromeDriverManager().install(),
            ) 
        table=[]
        for i in range(0,70):
               
                print(fundsUrl[i])
                driver.get(fundsUrl[i])
                rows= driver.find_element(By.TAG_NAME,"tbody")
                tr_tags =rows.find_elements(By.TAG_NAME,"tr")
                for tr in tr_tags:
                   cols=[]
                   td_tags=tr.find_elements(By.TAG_NAME,'td')
                   driver.execute_script("arguments[0].scrollIntoView()",tr)

                   for col in td_tags:
                       cols.append(col.text) 
                       name = [] 
                       name.append(fundsUrl[i].split("/")[-1])
                   table.append(cols+name)
        driver.close()
        file_name = "Ticker"
        df = pd.DataFrame(table)
        date = str(datetime.date.today())
        df.columns = ["S.No.", "Stock Name", "q1", "q2", "q3", "q4","q5","q6","Value"]
        df.to_csv(funds_path + "/{}.csv".format(file_name), index=None)


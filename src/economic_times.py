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


def get_stories(n = 10):
    driver = webdriver.Chrome(
    ChromeDriverManager().install(),
    )
    url = "https://economictimes.indiatimes.com/markets/stocks/recos"
    driver.get(url)
    sleep(10)
    
    for i in tqdm(range(n)):
        # print(i)
        try:
            close = driver.find_element(By.CLASS_NAME, "imgClose")
            close.click()
        except:
            pass
        sleep(1)
        auto_load = driver.find_element(By.CLASS_NAME, "autoload_continue")
        driver.execute_script("arguments[0].scrollIntoView();", auto_load)
        driver.execute_script("arguments[0].click();", auto_load)
    # sleep(200)
    res_stories = {
        "title":[],
        "link":[],
        "time":[],
        "text":[]
    }
    
    stories = driver.find_elements(By.CLASS_NAME, "eachStory")
    for story in tqdm(stories):
        link = story.find_element(By.TAG_NAME, "a").get_attribute("href")
        title = story.find_element(By.TAG_NAME, "a").text
        time = story.find_element(By.TAG_NAME, "time")
        p = story.find_element(By.TAG_NAME, "p").text
        html = story.get_property("innerHTML")
        soup = BeautifulSoup(html, "html.parser")
        time_tag = str(soup.find("time"))
        time = time_tag.split('"')[3]
        res_stories["title"].append(title)
        res_stories["link"].append(link)
        res_stories["time"].append(time)
        res_stories["text"].append(p)
    driver.close()
    df = pd.DataFrame(res_stories)
    df = df.fillna(" ")
    df.to_csv("data/news.csv", index=None)

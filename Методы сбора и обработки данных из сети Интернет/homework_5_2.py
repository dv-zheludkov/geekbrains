# 2) Написать программу, которая собирает «Новинки» с сайта техники mvideo и складывает данные в БД. Сайт можно выбрать
# и свой. Главный критерий выбора: динамически загружаемые товары

from pprint import pprint
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from pymongo import MongoClient
import json
import re

def add_to_db(db, *goods):
    n = 0
    for good in goods:
        item = json.loads(re.sub(r'" ', r'\" ', good.get_attribute('data-product-info')))
        db.update_one({'productId': item.get('productId')}, {'$set': item}, upsert=True)
        n += 1
    print("Добавлено товаров", n)


chrome_options = Options()
chrome_options.add_argument('start-maximized')
driver = webdriver.Chrome(executable_path='./chromedriver',options=chrome_options)
driver.get('https://www.mvideo.ru/')

new_items = driver.find_element_by_xpath("//h2[contains(text(), 'Новинки')]/ancestor::div[@class='section']")
actions = ActionChains(driver)
actions.move_to_element(new_items)
actions.perform()

time.sleep(3)
new_items.find_element_by_xpath(".//a[contains(@class, 'next-btn')]").click()
goods_check = len(new_items.find_elements_by_xpath(".//li//div[@class='fl-product-tile__picture-holder c-product-tile-picture__holder']/a"))
while True:
    time.sleep(5)
    new_items.find_element_by_xpath(".//a[contains(@class, 'next-btn')]").click()
    goods = new_items.find_elements_by_xpath(
        ".//li//div[@class='fl-product-tile__picture-holder c-product-tile-picture__holder']/a")
    if goods_check == len(goods):
        break
    goods_check = len(goods)


client = MongoClient('localhost', 27017)
db = client['db_goods']
dbg = db.dbg

add_to_db(dbg, *goods)

driver.close()
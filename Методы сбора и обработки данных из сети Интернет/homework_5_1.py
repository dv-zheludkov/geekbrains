# 1) Написать программу, которая собирает входящие письма из своего или тестового почтового ящика и сложить
# данные о письмах в базу данных (от кого, дата отправки, тема письма, текст письма полный)
# Логин тестового ящика: study.ai_172@mail.ru
# Пароль тестового ящика: NextPassword172!!!

from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from pymongo import MongoClient

driver = webdriver.Chrome(executable_path='./chromedriver')
driver.get("https://mail.ru")
login = driver.find_element_by_name('login')
login.send_keys('study.ai_172@mail.ru')
login.send_keys(Keys.ENTER)
time.sleep(4)
passw = driver.find_element_by_name('password')
passw.send_keys('NextPassword172!!!')
passw.send_keys(Keys.ENTER)
time.sleep(5)
def get_mail_links(driver):
    mails_links = set()
    while True:
        len_start = len(mails_links)
        mails = driver.find_elements_by_class_name('js-letter-list-item')
        for item in mails:
            mails_links.add(item.get_attribute('href'))
        len_end = len(mails_links)
        if len_end == len_start:
            break
        actions = ActionChains(driver)
        actions.move_to_element(mails[-1])
        # Имитирует прокрутку колесом мыши, раскоментировать если переход проходит только один раз
        # driver.execute_script("arguments[0].scrollIntoView()", mails[-1])
        actions.perform()
        time.sleep(5)
    return mails_links

def get_mails(mail_links, driver):
    mails = []

    for item in mail_links:
        mail_item = {}
        driver.get(item)
        time.sleep(7)
        mail_item['from'] = driver.find_element_by_xpath("//div[@class='letter__author']/span[@class='letter-contact']").get_attribute('title')
        mail_item['data'] = driver.find_element_by_xpath("//div[@class='letter__date']").text
        mail_item['subject'] = driver.find_element_by_xpath("//h2").text
        mail_item['text'] = driver.find_element_by_xpath("//div[@class='letter__body']").text
        mails.append(mail_item)
    return mails

def db_mails_add(db, *mails):
    n = 0
    for item in mails:
        db.insert_one(item)
        n += 1

    print(f"Писем добавлено  {n}")

mail_links = get_mail_links(driver)
mails = get_mails(mail_links, driver)


client = MongoClient('127.0.0.1', 27017)
db = client['mails_db']
dbm = db.mails

db_mails_add(dbm, *mails)


driver.close()

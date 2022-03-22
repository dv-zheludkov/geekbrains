# 1. Написать приложение, которое собирает основные новости с сайта на выбор news.mail.ru, lenta.ru, yandex-новости. Для парсинга использовать XPath. Структура данных должна содержать:
# название источника;
# наименование новости;
# ссылку на новость;
# дата публикации.
# 2. Сложить собранные данные в БД


from lxml import html
import requests
from fake_headers import Headers
from pymongo import MongoClient

headers = Headers().generate()

def get_news_mailru():
    news_mail = []
    response = requests.get('https://news.mail.ru', headers=headers)
    dom = html.fromstring(response.text)
    news_photo = dom.xpath("//div[contains(@class, 'daynews__item')]")
    mnews = []
    for news in news_photo:
        news_link = news.xpath(".//a/@href")
        mnews.append(news_link)
    news_txt = dom.xpath("//li[@class='list__item']")
    for news in news_txt:
        news_link = news.xpath(".//a/@href")
        mnews.append(news_link)
    for link in mnews:
        response_link = requests.get(link[0], headers=headers)
        dom = html.fromstring(response_link.text)
        news_page = dom.xpath("//div[@class='article js-article js-module']")
        for news in news_page:
            news_dict = {}
            news_dict['sourse'] = news.xpath(".//span[@class='link__text']/text()")[0]
            news_dict['title'] = news.xpath(".//h1/text()")[0].replace("\xa0"," ")
            news_dict['link'] = link[0]
            news_dict['date'] = news.xpath(".//span/@datetime")[0]
            news_mail.append(news_dict)
    return news_mail

def get_news_lentaru():
    news_lenta = []
    response = requests.get('https://lenta.ru/', headers=headers)
    dom = html.fromstring(response.text)
    news_dom = dom.xpath("//section[contains(@class,'for-main')]//a[not(@class)]")

    for news in news_dom:
        news_dict = {}
        news_dict['sourse'] = "Lenta.ru"
        news_dict['title'] = news.xpath(".//text()")[1].replace("\xa0"," ")
        news_dict['link'] = "https://lenta.ru/" + news.xpath("./@href")[0]
        news_dict['date'] = news.xpath(".//time/@datetime")[0]
        news_lenta.append(news_dict)
    return news_lenta

def get_news_yandexnews():
    news_lenta = []
    response = requests.get('https://yandex.ru/news/', headers=headers)
    dom = html.fromstring(response.text)
    news_dom = dom.xpath("//article[contains(@class,'mg-card')]")

    for news in news_dom:
        news_dict = {}
        news_dict['sourse'] = news.xpath(".//a[@class='mg-card__source-link']/text()")[0]
        news_dict['title'] = news.xpath(".//h2/text()")[0].replace("\xa0"," ")
        news_dict['link'] = news.xpath(".//a[@class='mg-card__link']/@href")[0]
        news_dict['date'] = news.xpath(".//span[@class='mg-card-source__time']/text()")[0]
        news_lenta.append(news_dict)
    return news_lenta

def db_news_add_new(db, *news_block):
    n = 0
    for news in news_block:
        if db.find_one({'link' : news['link']}):
            continue
        else:
            db.insert_one(news)
            n += 1

    print(f"Новостей добавлено  {n}")

news_block = []
news_block.extend(get_news_mailru())
news_block.extend(get_news_lentaru())
news_block.extend(get_news_yandexnews())

client = MongoClient('127.0.0.1', 27017)
db = client['news_db']
news = db.news

db_news_add_new(news, *news_block)



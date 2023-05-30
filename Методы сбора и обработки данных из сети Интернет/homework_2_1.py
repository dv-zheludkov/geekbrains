# Вариант 1
# Необходимо собрать информацию о вакансиях на вводимую должность (используем input или через аргументы)
# с сайтов Superjob и HH. Приложение должно анализировать несколько страниц сайта (также вводим через input
# или аргументы). Получившийся список должен содержать в себе минимум:
# Наименование вакансии.
# Предлагаемую зарплату (отдельно минимальную и максимальную).
# Ссылку на саму вакансию.
# Сайт, откуда собрана вакансия. ### По желанию можно добавить ещё параметры вакансии (например, работодателя
# и расположение). Структура должна быть одинаковая для вакансий с обоих сайтов. Общий результат можно вывести
# с помощью dataFrame через pandas.

from bs4 import BeautifulSoup as bs
import requests
import re
import pandas as pd


def _parser_hh(vacancy):
    vacancy_date = []

    params = {
        'text': vacancy, \
        'search_field': 'name', \
        'items_on_page': '100', \
        'page': ''
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0'
    }

    link = 'https://hh.ru/search/vacancy'

    html = requests.get(link, params=params, headers=headers)

    if html.ok:
        parsed_html = bs(html.text, 'html.parser')

        page_block = parsed_html.find('div', {'data-qa': 'pager-block'})
        if not page_block:
            last_page = '1'
        else:
            last_page = int(page_block.find_all('a', {'data-qa': 'pager-page'})[-2].getText())

    for page in range(0, last_page):
        params['page'] = page
        html = requests.get(link, params=params, headers=headers)

        if html.ok:
            parsed_html = bs(html.text, 'html.parser')

            vacancy_items = parsed_html.find('div', {'data-qa': 'vacancy-serp__results'}) \
                .find_all('div', {'class': 'vacancy-serp-item'})

            for item in vacancy_items:
                vacancy_date.append(_parser_item_hh(item))

    return vacancy_date


def _parser_item_hh(item):
    vacancy_date = {}
    vacancy_name = item.find('a', {'data-qa': 'vacancy-serp__vacancy-title'}) \
        .getText()
    vacancy_date['vacancy_name'] = vacancy_name
    company_name = item.find('div', {'class': 'vacancy-serp-item__meta-info-company'}) \
        .getText() \
        .replace(u' ', u' ' \
        .replace(u'<!-- -->', u' '))
    vacancy_date['company_name'] = company_name
    city = item.find('span', {'data-qa': 'vacancy-serp__vacancy-address'}) \
        .getText() \
        .split(', ')[0]
    vacancy_date['city'] = city
    salary = item.find('span', {'data-qa': 'vacancy-serp__vacancy-compensation'})

    if not salary:
        salary_min = None
        salary_max = None
        salary_currency = None
    else:
        salary = salary.getText() \
            .replace(u' ', u'').replace(u' ', u' ')
        salary = re.split('-', salary)
        if len(salary) == 1:
            spam = re.split(' ', salary[0])
            if spam[0] == 'до':
                salary_min = None
                salary_max = "".join(map(str, spam[1])) if len(spam[1]) > 1  else spam[1]
                salary_currency = "".join(map(str, spam[2:])) if len(spam[2:]) > 1  else spam[2:]
            elif spam[0] == 'от':
                salary_min = "".join(map(str, spam[1])) if len(spam[1]) > 1  else spam[1]
                salary_max = None
                salary_currency = "".join(map(str, spam[2:])) if len(spam[2:]) > 1  else spam[2:]
            else:
                salary_min = "По договорённости"
                salary_max = "По договорённости"
                salary_currency = "-"
        else:
            spam1 = re.split(' ', salary[0])
            spam2 = re.split(' ', salary[1])
            salary_min = "".join(map(str, spam1[0])) if len(spam1[0]) > 1  else spam1[0]
            salary_max = "".join(map(str, spam2[0])) if len(spam2[0]) > 1  else spam2[0]
            salary_currency = "".join(map(str, spam2[1:])) if len(spam2[1:]) > 1  else spam2[1:]

    vacancy_date['salary_min'] = salary_min
    vacancy_date['salary_max'] = salary_max
    vacancy_date['salary_currency'] = salary_currency

    # link
    vacancy_link = item.find('div', {'class': 'vacancy-serp-item__row vacancy-serp-item__row_header'}) \
        .find('a')['href']

    vacancy_link = vacancy_link.split('?')[0]

    vacancy_date['vacancy_link'] = vacancy_link

    # site
    vacancy_date['site'] = 'www.hh.ru'

    return vacancy_date


def _parser_superjob(vacancy):
    vacancy_date = []

    params = {
        'keywords': vacancy, \
        'profession_only': '1', \
        'geo[c][0]': '15', \
        'geo[c][1]': '1', \
        'geo[c][2]': '9', \
        'page': ''
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0'
    }

    link = 'https://www.superjob.ru/vacancy/search/'

    html = requests.get(link, params=params, headers=headers)

    if html.ok:
        parsed_html = bs(html.text, 'html.parser')

        page_block = parsed_html.find('a', {'class': 'f-test-button-1'})
    if not page_block:
        last_page = 1
    else:
        page_block = page_block.findParent()
        last_page = int(page_block.find_all('a')[-2].getText())

    for page in range(0, last_page + 1):
        params['page'] = page
        html = requests.get(link, params=params, headers=headers)

        if html.ok:
            parsed_html = bs(html.text, 'html.parser')
            vacancy_items = parsed_html.find_all('div', {'class': 'f-test-vacancy-item'})

            for item in vacancy_items:
                vacancy_date.append(_parser_item_superjob(item))

    return vacancy_date


def _parser_item_superjob(item):
    vacancy_date = {}
    vacancy_name = item.find_all('a')
    if len(vacancy_name) > 1:
        vacancy_name = vacancy_name[-2].getText()
    else:
        vacancy_name = vacancy_name[0].getText()
    vacancy_date['vacancy_name'] = vacancy_name
    company_name = item.find('span', {'class': 'f-test-text-vacancy-item-company-name'})
    if not company_name:
        company_name = item.findParent() \
                            .find('span', {'class': 'f-test-text-vacancy-item-company-name'})
    if not company_name:
        company_name = "-"
    else:
        company_name = company_name.getText()
    vacancy_date['company_name'] = company_name
    company_location = item.find('span', {'class': 'f-test-text-company-item-location'}) \
        .findChildren()[1] \
        .getText() \
        .split(',')
    vacancy_date['city'] = company_location[0]
    salary = item.find('span', {'class': 'f-test-text-company-item-salary'})
    if not salary:
        salary_min = None
        salary_max = None
        salary_currency = None
    else:
        salary = salary.getText() \
            .replace(u' ', u'').replace(u' ', u' ')
        salary = re.split('—', salary)
        if len(salary) == 1:
            spam = re.split(' ', salary[0])
            if spam[0] == 'до':
                salary_min = None
                salary_max = "".join(map(str, spam[1:3])) if len(spam[1:3]) > 1  else spam[1:3]
                salary_currency = "".join(map(str, spam[3:])) if len(spam[3:]) > 1  else spam[3:]
            elif spam[0] == 'от':
                salary_min = "".join(map(str, spam[1:3])) if len(spam[1:3]) > 1  else spam[1:3]
                salary_max = None
                salary_currency = "".join(map(str, spam[3:])) if len(spam[3:]) > 1  else spam[3:]
            else:
                salary_min = "По договорённости"
                salary_max = "По договорённости"
                salary_currency = "-"
        else:
            spam1 = re.split(' ', salary[0])
            spam2 = re.split(' ', salary[1])
            salary_min = "".join(map(str, spam1[0:2])) if len(spam1[0:2]) > 1  else spam1[0:2]
            salary_max = "".join(map(str, spam2[0:3])) if len(spam2[0:3]) > 1  else spam2[0:3]
            salary_currency = "".join(map(str, spam2[3:])) if len(spam2[3:]) > 1  else spam2[3:]
    vacancy_date['salary_min'] = salary_min
    vacancy_date['salary_max'] = salary_max
    vacancy_date['salary_currency'] = salary_currency
    vacancy_link = item.find_all('a')
    if len(vacancy_link) > 1:
        vacancy_link = vacancy_link[-2]['href']
    else:
        vacancy_link = vacancy_link[0]['href']
    vacancy_date['vacancy_link'] = f'https://www.superjob.ru{vacancy_link}'
    vacancy_date['site'] = 'www.superjob.ru'
    return vacancy_date


def parser_vacancy(vacancy):
    vacancy_date = []
    vacancy_date.extend(_parser_hh(vacancy))
    vacancy_date.extend(_parser_superjob(vacancy))

    df = pd.DataFrame(vacancy_date)

    return df

vacancy = 'Преподаватель'
df = parser_vacancy(vacancy)
pd.set_option('display.max_rows', 12)

print(df)

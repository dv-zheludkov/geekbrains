# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from pymongo import MongoClient
import re

class JobparserPipeline:
    def __init__(self):
        client = MongoClient('localhost', 27017)
        self.mongo_base = client.vacancies1708

    def process_item(self, item, spider):

        if spider.name == 'hhru':
            item['salary'] = self.process_salary_hh(item['salary'])
        if spider.name == 'sjru':
            item['salary'] = self.process_salary_sj(item['salary'])
        collection = self.mongo_base[spider.name]
        collection.insert_one(item)
        return item

    def process_salary_hh(self, salary):
        min_s = None
        max_s = None
        cur = None
        if salary != 'з/п не указана':
            salary = salary.replace(u' ', u'').replace(u' ', u'')
            salary = re.split('до ', salary)
            if len(salary) == 1:
                spam = re.split(' ', salary[0])
                min_s = int(spam[1])
                max_s = None
                cur = spam[2]
            elif salary[0] == '':
                spam = re.split(' ', salary[1])
                min_s = None
                max_s = int(spam[0])
                cur = spam[1]
            else:
                spam1 = re.split(' ', salary[0])
                spam2 = re.split(' ', salary[1])
                min_s = int(spam1[1])
                max_s = int(spam2[0])
                cur = spam2[1]

        return [min_s, max_s, cur]

    def process_salary_sj(self, salary):
        min_s = None
        max_s = None
        cur = None
        sss = salary
        if salary != 'По договорённости':
            salary = salary.replace(u' ', u' ').replace(u' ', u' ')
            salary = re.split(' — ', salary)
            if len(salary) == 1:
                spam = re.split(' ', salary[0])
                if spam[0] == 'от':
                    min_s = int(spam[1] + spam[2])
                    max_s = None
                    cur = "".join(map(str, spam[3:]))
                elif spam[0] == 'до':
                    min_s = None
                    max_s = int(spam[1] + spam[2])
                    cur = "".join(map(str, spam[3:]))
                else:
                    min_s = int(spam[0] + spam[1])
                    max_s = int(spam[0] + spam[1])
                    cur = "".join(map(str, spam[2:]))
            else:
                spam1 = re.split(' ', salary[0])
                spam2 = re.split(' ', salary[1])
                min_s = int(spam1[0] + spam1[1])
                max_s = int(spam2[0] + spam2[1])
                cur = "".join(map(str, spam2[2:]))
        print(sss, "\n", salary, "\n", min_s, max_s, cur)

        return [min_s, max_s, cur]
import scrapy
from scrapy.http import HtmlResponse
from jobparser.items import JobparserItem
import w3lib.html

class SjruSpider(scrapy.Spider):
    name = 'sjru'
    allowed_domains = ['superjob.ru']
    start_urls = ['https://russia.superjob.ru/vacancy/search/?keywords=python']

    def parse(self, response: HtmlResponse):
        links = response.xpath("//div[contains(@class,'f-test-vacancy-item')]/div/div/div//div/div/a/@href").extract()

        next_page = response.xpath("//a[contains(@class,'f-test-link-Dalshe')]/@href").extract_first()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

        for link in links:
            yield response.follow(link, callback=self.vacancy_parse)


    def vacancy_parse(self, response: HtmlResponse):
        name_data = response.xpath("//h1/text()").extract_first()
        link_data = response.url
        salary_data = response.xpath("//div[contains(@class,'f-test-vacancy-base-info')]/div/div/div/div/span/span").extract_first()
        salary_data = w3lib.html.remove_tags(salary_data)
        yield JobparserItem(name=name_data, salary=salary_data, link=link_data, site='superjob.ru')



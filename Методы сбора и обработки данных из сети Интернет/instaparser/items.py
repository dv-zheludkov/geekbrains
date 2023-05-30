# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class InstaparserItem(scrapy.Item):
    # define the fields for your item here like:
    target_user = scrapy.Field()
    target_user_id = scrapy.Field()
    f_user = scrapy.Field()
    _id = scrapy.Field()
    f_photo_link = scrapy.Field()
    target_user_to_f_user_relation = scrapy.Field()






# 2. Изучить список открытых API. Найти среди них любое, требующее авторизацию (любого типа).
# Выполнить запросы к нему, пройдя авторизацию. Ответ сервера записать в файл.

import requests

url = 'https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/'
token = 'pk.eyJ1IjoiZHYtemhlbHVka292IiwiYSI6ImNrcXo0Nmk5azB6N24ydnBsMTA2MG45dzkifQ.dHEr62wpzsrI2Ysx06hhug'

r = requests.get(f'{url}-122.4241,37.78,15.25,0,60/400x400?access_token={token}')

with open('homework_1_2_pic.png', 'bw') as f:
    f.write(r.content)
with open('homework_1_2_headers.txt', 'w') as f:
    f.write(str(r.headers))


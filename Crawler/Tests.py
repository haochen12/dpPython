from urllib import request, error
import requests as rq
import lxml.etree as et

try:
    for i in range(10):
        url = "https://maoyan.com/board/4?offset={}".format(i * 10)
        print(url)
        test = rq.get(url)
        t = et.HTML(test.text)
        print(t.xpath('//*[@id="app"]/div/div/div[1]/dl/dd/a/@title'))
        print(t.xpath('//*[@id="app"]/div/div/div[1]/dl/dd/div/div/div[1]/p/text()'))

except error.HTTPError as e:
    print(e.reason)

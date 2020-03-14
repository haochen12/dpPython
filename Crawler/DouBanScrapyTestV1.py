from __future__ import division
from lxml import etree
import requests as rq
import re
import numpy as np

import matplotlib.pyplot as plt

source_url = 'http://movie.douban.com/top250/'
url_form = 'http://movie.douban.com/top250/%s'

all_item1 = []


# 获取网页内容
def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'
    }
    return rq.get(url, headers).content


def parse_page(content):
    html = etree.HTML(content)
    pages = html.xpath('//*[@id="content"]/div/div[1]/div[@class="paginator"]/a/@href')  # 获取页码
    return pages


def parse_item(content):
    html = etree.HTML(content)
    grid_view = html.xpath('//*[@id="content"]/div/div[1]/ol/li')

    for i in range(len(grid_view)):
        item = []
        # title
        image_url = '//li[%d]/div[@class="item"]/div[@class="pic"]/a/img/@src' % (i + 1)
        image_url_list = grid_view[i].xpath(image_url)
        item.append(image_url_list[0])
        # print("image_url:", image_url_list)

        # item_url
        item_url = '//li[%d]/div[@class="item"]/div[@class="info"]/div[@class="hd"]/a/@href' % (i + 1)
        item_url_list = grid_view[i].xpath(item_url)
        item.append(item_url_list[0])
        # print("item_url:", item_url_list)

        # image_title
        item_title = '//li[%d]/div[@class="item"]/div[@class="info"]/div[@class="hd"]/a/span[@class="title"]' \
                     '/text()' % (i + 1)
        item_title_list = grid_view[i].xpath(item_title)
        item.append(item_title_list[0])
        # print("item_title:", item_title_list)

        # item_other
        item_other = '//li[%d]/div[@class="item"]/div[@class="info"]/div[@class="hd"]/a/span[@class="other"]' \
                     '/text()' % (i + 1)
        item_other_list = grid_view[i].xpath(item_other)
        item.append(item_other_list[0])
        # print("item_other:", item_other_list)

        # item_score
        item_score = '//li[%d]/div[@class="item"]/div[@class="info"]/div[@class="bd"]/div[@class="star"]' \
                     '/span/text()' % (i + 1)
        item_score_list = grid_view[i].xpath(item_score)
        item.append(float(item_score_list[0]))
        item.append(item_score_list[1])
        # print('---------------------------splitter--------------------------------------')
        # print(item)
        all_item1.append(item)
    return all_item1


def main():
    parse_item(get_html(source_url))
    page_list = parse_page(get_html(source_url))
    for i in page_list:
        url = url_form % i
        parse_item(get_html(url))
    # print(all_item1)


def rex_digital(string):
    s = '^[0-9]\d*'
    return float(re.match(s, string, re.M | re.I).group(0))


if __name__ == '__main__':
    main()
    item_test = []
    for i in range(250):
        item_test.append([all_item1[i][3], all_item1[i][4], rex_digital(all_item1[i][5])])
    test = np.asarray(item_test)
    ax = plt.subplot(111)
    # print(test[:, 0])
    # plt.hist2d(test[:, 0], test[:, 1], int=10,)
    plt.scatter(test[:, 1], test[:, 2])
    for z, k, v in zip(test[:, 0], test[:, 1], test[:, 2]):
        plt.annotate(z, xy=(k, v), xytext=(k, v))
    plt.show()

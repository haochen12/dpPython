from lxml import etree
import requests as rq

source_url = 'http://movie.douban.com/top250/'
url_form = 'http://movie.douban.com/top250/%s'


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
        image_url_list = grid_view[i].xpath(image_url);
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
        item.append(item_score_list[0])
        item.append(item_score_list[1])

        # print("item_score:", item_score_list)

        print('---------------------------splitter--------------------------------------')
        print(item)
        # return all_item_list


def main():
    parse_item(get_html(source_url))
    page_list = parse_page(get_html(source_url))
    for i in page_list:
        url = url_form % i
        parse_item(get_html(url))


main()

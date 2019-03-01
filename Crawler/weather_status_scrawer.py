import requests

import lxml.html

url = 'https://blog.csdn.net/itachi85/article/details/50773358'


def get_html_content():
    response = requests.get(url)
    print("----------get html content----------")
    return response.text


def parse_html():
    print("----------parse html----------")
    context_string = get_html_content()
    root = lxml.html.fromstring(context_string)

    # print(root.xpath('//*[@id="content"]/div[class="api_month_list"]/[tag="text"'))
    print(root.xpath('//*[@id="asideProfile"]/div[2]/dl/dt/text()'))
    print(root.xpath('//*[@id="asideProfile"]/div[@class="data-info d-flex item-tiling"]'))

def get_weather_conditions():
    print("skjfldkjldkfjslkdjf")


def get_months_list():
    print()


if __name__ == '__main__':
    parse_html()

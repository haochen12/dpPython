import Crawler.DouBanScrapyTestV1 as test
import Crawler.DBManager as dbTEst
from lxml import etree

commits_url = 'https://movie.douban.com/subject/1292052/comments?start=60&limit=20&sort=new_score&status=P'
t = test.get_html(commits_url)


def parse_commits(content):
    # db = dbTEst.DBManager()
    html = etree.HTML(content)
    xpath_string = '//*[@id="comments"]/div[@class="comment-item"]/div[@class="comment"]'
    commits_text = html.xpath(xpath_string + '//p/span/text()')
    commits_use_count = html.xpath(xpath_string + '//h3/span[1]/span/text()')
    commits_user_name = html.xpath(xpath_string + '//h3/span[2]/a/text()')
    commits_date = html.xpath(xpath_string + '//h3/span[2]/span[3]/@title')
    commits_score = html.xpath(xpath_string + '//h3/span[2]/span[2]/@title')

    for (k, j, m, n, d) in zip(commits_text, commits_score, commits_use_count, commits_user_name, commits_date):
        print(k, j, m, n, d)
        # db.insert_data1(k, j, m, n, d)


page = 0

commits_url_form = 'https://movie.douban.com/subject/1292052/comments?start=%d&limit=20&sort=new_score&status=P'
while True:
    temp_url = commits_url_form % page
    page = page + 20
    result = test.get_html(temp_url)
    print(temp_url)
    parse_commits(result)
    if page == 200:
        break

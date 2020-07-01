from __future__ import division
from lxml import etree
import requests as rq
import re
import numpy as np

import json

url = "https://voice.baidu.com/act/newpneumonia/newpneumonia"

# 获取网页内容
def get_html(url):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}

    src_html = rq.get(url, headers).content
    return src_html


def parse_page(content):
    html = etree.HTML(content)
    pages = html.xpath('/html/body/script[8]/text()')
    return pages


for str in parse_page(get_html(url)):
    d = json.loads(str, encoding="utf-8")

    component = d["component"]

    for objts in component[0]["caseList"]:
        print(objts["confirmed"],
              objts["died"],
              objts["crued"],
              objts["relativeTime"],
              objts["confirmedRelative"],
              objts["diedRelative"],
              objts["curedRelative"],
              objts["curConfirm"],
              objts["area"])

    for oj in component[0]["globalList"]:
        print(oj)


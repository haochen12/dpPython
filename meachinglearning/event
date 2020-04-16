# coding=utf-8

import xlrd
import xpinyin

# 打开文件
data = xlrd.open_workbook('20200415.xlsx')
p = xpinyin.Pinyin()
# 查看工作表
data.sheet_names()
print("sheets：" + str(data.sheet_names()))

# 通过文件名获得工作表,获取工作表1
table = data.sheet_by_name('安卓4.0教师端埋点')

rows = table.nrows

for i in range(1, rows):
    test = table.row_values(i)
    # print(test[3], test[5], test[4])
    print("public static final String ",
          p.get_initials(test[3], "").replace("（", "").replace("）", "") + "_" +
          p.get_initials(test[5], "").lstrip().replace('-', '_').replace('【',"").replace('】','') + "=",
          "\""+str(test[4]).replace(".0", "")+"\"" + ";//" + test[3] + test[5])
#

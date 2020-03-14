import Crawler.DBManager as dbTEst
import matplotlib.pyplot as plt
import numpy as np

db_test = dbTEst.DBManager()
datasets = db_test.select_data()

list_dataset = list(datasets)

X = []
Y = []

for data_tuple in list_dataset:
    X.append(data_tuple[5])
    Y.append(data_tuple[3])

Y_test = np.asarray(Y, float)
plt.plot(X, np.asarray(Y, float))
year = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
plt.xticks(np.arange(0, 196, 14), year)
plt.yticks(np.arange(0, np.max(Y_test), 1000), np.arange(np.min(Y_test), np.max(Y_test), 500))
plt.show()

# 很差，较差，还行，推荐，力荐

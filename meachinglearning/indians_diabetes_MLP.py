from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow

import numpy as np

np.random.seed(7)

# 1.load data 加载数据
dataset = np.loadtxt("E:\pythonProject\data\pima-indians-diabetes.csv", delimiter=',')

Y = dataset[:, 8]
X = dataset[:, 0:8]

# 2.create model 创建模型

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3.compile model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=['accuracy'])
# 4.fit model 拟合模型
model.fit(x=X, y=Y, validation_split=0.30, batch_size=10, epochs=500)

# scores = model.predict(x=X)
# print(scores)

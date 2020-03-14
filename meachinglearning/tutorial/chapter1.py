import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(1, activation=tf.keras.activations.linear))

model.compile(optimizer="SGD", loss="mean_squared_error", metrics=['accuracy'])

X = np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float)
Y = np.asarray([1, 0, 0, 0], dtype=np.float)
import matplotlib.pyplot as plt

history = model.fit(x=X,
                    y=Y,
                    batch_size=2,
                    epochs=1200)
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print(model.predict(x=X))

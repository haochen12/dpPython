# encoding:utf-8

# three layer neural network
import tensorflow as tf
from tensorflow import keras
import numpy as np

# (60000, 28, 28)
(x_train, y_train) = np.array([[[1, 1]],
                               [[1, 0]],
                               [[0, 1]],
                               [[0, 0]]]), np.array([1, 0, 0, 0])
print(x_train.shape)

model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(1, 2), units=1))
model.add(keras.layers.Dense(1, activation=keras.activations.linear))

model.compile(optimizer=keras.optimizers.SGD.__name__,
              loss=keras.losses.mean_squared_error.__name__,
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100)
print(model.predict([[[0, 1]]]))
model.summary()

# model.evaluate(x_test, y_test, verbose=2)
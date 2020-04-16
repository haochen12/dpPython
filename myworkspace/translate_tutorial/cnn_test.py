from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, train_labels = train_images / 255.0, train_labels / 255.0
x_train4D = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
x_test4D = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

model = models.Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 input_shape=(28, 28, 1),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(Conv2D(filters=32,
                 kernel_size=(5, 5),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(5, 5),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,
                 kernel_size=(5, 5),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x=x_train4D, y=to_categorical(train_labels), batch_size=100, epochs=2)

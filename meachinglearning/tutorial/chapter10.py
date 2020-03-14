from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pandas import read_csv
import tensorflow as tf
import numpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

# load data
dataframe = read_csv('data/iris.data', header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# 向量转换
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)
dummy_Y = np_utils.to_categorical(encoder_Y)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation=tf.keras.activations.relu))
    model.add(Dense(3, activation=tf.keras.activations.sigmoid))

    # compile model
    model.compile(loss=tf.losses.categorical_crossentropy, optimizer="adam",
                  metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=1500, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print(results.mean())

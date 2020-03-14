from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import numpy


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation=tf.keras.activations.relu))
    model.add(Dense(8, activation=tf.keras.activations.relu))
    model.add(Dense(1, activation=tf.keras.activations.sigmoid))

    # compile model
    model.compile(loss=tf.losses.binary_crossentropy, optimizer=tf.optimizers.Adam(0.1), metrics=['accuracy'])
    return model


seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("E:\pythonProject\data\pima-indians-diabetes.csv", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=1)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)

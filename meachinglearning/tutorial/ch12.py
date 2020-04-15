import numpy as np
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load data

data_frame = read_csv("E:/pythonProject/meachinglearning/tutorial/data/housing.csv", delim_whitespace=True, header=None)
dataset = data_frame.values

# split data into input(X) and output(Y)
X = dataset[:, 0:13]
Y = dataset[:, 13]


def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# evaluate model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=1)))

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, random_state=seed)
pipeline = Pipeline(estimators)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Baseline:%.2f(%.2f) MSE" % (results.mean(), results.std()))

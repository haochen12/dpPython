import numpy
from pandas import read_csv
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = read_csv("E:/pythonProject/meachinglearning/tutorial/data/sonar.csv", header=None)
datasets = dataframe.values
# print(datasets)

# split into input(X) and output(Y) variables

X = datasets[:, 0:60].astype(float)
Y = datasets[:, 60]

# encode class values as integers

encoder = LabelEncoder()
encoder.fit(Y)
encoder_y = encoder.transform(Y)


# print(encoder_y)

def create_baseline():
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation="relu"))
    # model.add(Dense(30, kernel_initializer='normal', activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 方法1：
# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoder_y, cv=kfold)
#
# print("Baseline:%.2f%%(%.2f%%)" % (results.mean() * 100, results.std() * 100))

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoder_y, cv=kfold)

print("Baseline:%.2f%%(%.2f%%)" % (results.mean() * 100, results.std() * 100))



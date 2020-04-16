from sklearn.datasets import make_circles
from tensorflow.keras.layers import Dense, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot

X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# print(X, y)
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

model = Sequential()
model.add(Dense(50, input_dim=2, activation="relu", kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
# model.add(GaussianNoise(stddev=0.01))
model.add(Dense(1, activation="sigmoid"))
opt = SGD(lr=.01, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
history = model.fit(trainX,
                    trainy,
                    validation_data=(testX, testy),
                    epochs=1000,
                    batch_size=len(trainX),
                    verbose=1, callbacks=[es])

_, train_acc = model.evaluate(trainX, trainy, verbose=1)
_, test_acc = model.evaluate(testX, testy, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.grid()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.grid()
pyplot.show()

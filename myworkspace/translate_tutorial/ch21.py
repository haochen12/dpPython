# Plot ad hoc CIFAR10 instances
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot

from PIL import Image

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(Image.fromarray(X_train[i]))
    # show the plot
pyplot.show()

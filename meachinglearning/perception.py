import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, input_num):
        self.weights = np.random.random(input_num)
        self.bias = 0

    def __str__(self):
        print("weights", self.weights)

    def train(self, train_datas, labels):
        for x, y in zip(train_datas, labels):
            self.update_weights(x, y)

        print("train........")

    def update_weights(self, train_data, label):
        delta = label - self.predict(train_data)
        self.weights += self.weights * delta * 0.1
        self.bias += delta * 0.1

        print("update_weights", delta, self.weights, self.bias)

    def predict(self, train_data):
        print("predict")
        result = np.dot(train_data, self.weights) + self.bias
        return self.activation(result)

    def activation(self, x):
        return 1 if x > 0 else 0


if __name__ == "__main__":
    perceptron = Perceptron(2)
    for i in range(20):
        perceptron.train([[1, 1], [1, 0], [0, 1], [0, 0]], [1, 0, 0, 0])

    print(perceptron.predict([1, 1]))
    print(perceptron.predict([0, 1]))
    print(perceptron.predict([1, 0]))
    print(perceptron.predict([0, 0]))

    plt.scatter(x=[1, 0, 1, 0], y=[1, 1, 0, 0], c=[1, 0, 0, 0])
    plt.show()

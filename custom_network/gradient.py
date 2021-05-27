import sys

class Gradient:

    def __init__(self, learning_rate, method = 'default') -> None:
        self.lr = learning_rate
        self.method = method

    def compute(self, weights, gradient):

        if self.method == 'default':
            return weights - self.lr * gradient
            
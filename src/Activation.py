import numpy as np

class relu:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    def backward(self, derivative):
        self.derivative = derivative.copy()
        self.derivative[self.input <= 0] = 0
        #To avoid arithmetic error we are using <=
class softmax:
    def forward(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        self.output = prob
    def backward(self, derivative):
        self.derivative = derivative.copy()

class sigmoid:
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
    def backward(self, derivative):
        self.derivative = derivative * ( 1 - self.output) * self.output

class linear:
    def forward(self, input):
        self.input = input
        self.output = input

    def backward(self, derivative):
        self.derivative = derivative.copy()


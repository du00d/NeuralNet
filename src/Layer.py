import numpy as np


class Dense:
    def __init__(self, input_size, num_neurons, reg_weight_l1=0, reg_weight_l2=0, reg_bias_l1=0, reg_bias_l2=0):
        self.weight = 0.01 * np.random.rand(input_size, num_neurons)
        self.bias = np.zeros((1, num_neurons))
        ##################
        # REGULARIZATION #
        ##################
        self.reg_weight_l1 = reg_weight_l1
        self.reg_weight_l2 = reg_weight_l2
        self.reg_bias_l1 = reg_bias_l1
        self.reg_bias_l2 = reg_bias_l2

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weight) + self.biases

    def backward(self, derivative):
        self.dweight = np.dot(self.input.T, derivative)
        self.dbias = np.sum(derivative, axis=0, keepdims=True)

        ##################
        # REGULARIZATION #
        ##################
        '''
        d|w|      {   1  w > 0
        -----  = { 
        d w       {   -1 w < 0

        Derivative for L1 regularization
        '''
        if self.reg_weight_l1 != 0:
            dl1 = self.weight.copy()
            dl1[dl1 >= 0] = 1
            dl1[dl1 < 0] = -1
            self.dweight += self.reg_weight_l1 * self.weight
        if self.reg_weight_l2 != 0:
            self.dweight += 2 * self.reg_weight_l2 * self.weight

        if self.reg_bias_l1 != 0:
            dl1 = self.bias.copy()
            dl1[dl1 >= 0] = 1
            dl1[dl1 < 0] = -1
            self.dbias += self.reg_bias_l2 * self.bias
        if self.reg_bias_l2 != 0:
            self.dbias += 2 * self.reg_bias_l2 * self.bias

        self.derivative = np.dot(derivative, self.weight.T)


class Dropout:
    # Only used in training
    # This is how many percent of the neuron are we going to turn off for one step
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, input):
        self.mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
        self.output = input * self.mask

    def backward(self, derivative):
        # Dropout derivative turns out to be the same!
        self.derivative = derivative * self.mask
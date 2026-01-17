import numpy as np

class Momentum:
    def __init__(self, parameters, learning_rate, beta):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.momentums = [np.zeros_like(p.value) for p in parameters]

    def step(self):
        for i in range(len(self.parameters)):
            self.momentums[i] = self.beta * self.momentums[i] + self.parameters[i].grad
            self.parameters[i].value = self.parameters[i].value - self.momentums[i] * self.learning_rate

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
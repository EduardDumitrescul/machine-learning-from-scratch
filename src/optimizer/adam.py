import numpy as np

class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.first_moments = [np.zeros_like(p.value) for p in self.parameters]
        self.second_moments = [np.zeros_like(p.value) for p in self.parameters]
        self.epsilon = 1e-9
        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.value)

    def step(self):
        self.t += 1
        for i in range(len(self.parameters)):
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * self.parameters[i].grad
            corrected_first_moment = self.first_moments[i] / (1 - self.beta1 ** self.t)

            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * np.square(self.parameters[i].grad)
            corrected_second_moment = self.second_moments[i] / (1 - self.beta2 ** self.t)

            self.parameters[i].value = self.parameters[i].value - self.learning_rate * corrected_first_moment / (np.sqrt(corrected_second_moment) + self.epsilon)
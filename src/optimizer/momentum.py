class Momentum:
    def __init__(self, parameters, learning_rate, beta):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = 0

    def step(self):
        for parameter in self.parameters:
            self.velocity = self.beta * self.velocity + parameter.grad
            parameter.value = parameter.value - self.velocity * self.learning_rate

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
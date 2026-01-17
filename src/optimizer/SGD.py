class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.value = parameter.value - parameter.grad * self.learning_rate

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
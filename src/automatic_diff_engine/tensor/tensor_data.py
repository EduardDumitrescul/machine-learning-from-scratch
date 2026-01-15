class TensorData:
    def __init__(self, value, grad, requires_grad):
        self.value = value
        self.grad = grad
        self.requires_grad = requires_grad
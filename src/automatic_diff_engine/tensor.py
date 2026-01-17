from typing import Self, Type

import numpy as np

from automatic_diff_engine.function import Function
from automatic_diff_engine.functions.addition_function import AdditionFunction
from automatic_diff_engine.functions.exponential_function import ExponentialFunction
from automatic_diff_engine.functions.log import Log
from automatic_diff_engine.functions.matrix_multiplication import MatrixMultiplication
from automatic_diff_engine.functions.mean_function import MeanFunction
from automatic_diff_engine.functions.multiplication_function import MultiplicationFunction
from automatic_diff_engine.functions.relu_function import ReLU
from automatic_diff_engine.functions.sigmoid import Sigmoid
from automatic_diff_engine.functions.subtraction_function import SubtractionFunction
from automatic_diff_engine.functions.sum import Sum
from automatic_diff_engine.tensor_data import TensorData


class Tensor:
    def __init__(
            self,
            data,
            creator_func: Type[Function] = None,
            creator_operands = None,
            requires_grad = True,
    ):
        data = np.atleast_1d(np.array(data, dtype=np.float64))
        grad = np.zeros_like(data, dtype=np.float64)
        self.data = TensorData(data, grad, requires_grad)
        self.creator_func = creator_func
        self.creator_operands = creator_operands if creator_operands is not None else []


    def backward(self):
        self.grad = np.ones_like(self.value, dtype=np.float64)

        topo: list[Self] = list()
        visited = set()
        def build_topo(node: Self) -> None:
            if node not in visited:
                visited.add(node)
                for next_node in node.creator_operands:
                    build_topo(next_node)
                topo.append(node)

        build_topo(self)
        for tensor in reversed(topo):
            if tensor.creator_func is None:
                continue
            grads = tensor.creator_func.backward(tensor.grad, *[t.data for t in tensor.creator_operands])
            if not isinstance(grads, tuple):
                grads = (grads,)

            for i in range(len(tensor.creator_operands)):
                if tensor.creator_operands[i].requires_grad:
                    if tensor.creator_operands[i].grad is None:
                        tensor.creator_operands[i].grad = grads[i]
                    else:
                        tensor.creator_operands[i].grad = tensor.creator_operands[i].grad + grads[i]

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.value, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Tensor: {self.data}"

    def __add__(self, other) -> Self:
        result = Tensor(
            data = AdditionFunction.forward(self.data, other.data),
            creator_func = AdditionFunction,
            creator_operands = [self, other]
        )

        return result

    def __sub__(self, other) -> Self:
        result = Tensor(
            data=SubtractionFunction.forward(self.data, other.data),
            creator_func=SubtractionFunction,
            creator_operands=[self, other]
        )

        return result

    def __mul__(self, other) -> Self:
        result = Tensor(
            data = MultiplicationFunction.forward(self.data, other.data),
            creator_func = MultiplicationFunction,
            creator_operands = [self, other]
        )

        return result

    def __pow__(self, power, modulo=None):
        power = Tensor(
            data = power,
            creator_func = None,
            creator_operands = [],
            requires_grad = False
        )
        result = Tensor(
            data = ExponentialFunction.forward(self.data, power.data),
            creator_func = ExponentialFunction,
            creator_operands = [self, power]
        )

        return result

    def __matmul__(self, other: Self) -> Self:
        result = Tensor(
            data =  MatrixMultiplication.forward(self.data, other.data),
            creator_func = MatrixMultiplication,
            creator_operands = [self, other]
        )
        return result

    def mean(self):
        result = Tensor(
            data = MeanFunction.forward(self.data),
            creator_func = MeanFunction,
            creator_operands = [self]
        )
        return result

    def relu(self) -> Self:
        result = Tensor(
            data = ReLU.forward(self.data),
            creator_func=ReLU,
            creator_operands = [self]
        )
        return result

    def sigmoid(self) -> Self:
        result = Tensor(
            data = Sigmoid.forward(self.data),
            creator_func = Sigmoid,
            creator_operands = [self]
        )
        return result

    def log(self) -> Self:
        result = Tensor(
            data = Log.forward(self.data),
            creator_func = Log,
            creator_operands = [self]
        )
        return result

    def sum(self) -> Self:
        result = Tensor(
            data = Sum.forward(self.data),
            creator_func = Sum,
            creator_operands = [self]
        )
        return result

    def __array__(self, dtype=None):
        if dtype:
            return self.value.astype(dtype)
        return self.value

    @property
    def value(self) -> np.typing.NDArray[np.float64]:
        return self.data.value

    @value.setter
    def value(self, value) -> None:
        self.data.value = value

    @property
    def grad(self) -> np.typing.NDArray[np.float64]:
        return self.data.grad

    @grad.setter
    def grad(self, grad) -> None:
        self.data.grad = grad

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad) -> None:
        self.data.requires_grad = requires_grad

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

from typing import Self, Type

import numpy as np

from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.functions.addition_function import AdditionFunction
from automatic_diff_engine.tensor.functions.exponential_function import ExponentialFunction
from automatic_diff_engine.tensor.functions.matrix_multiplication import MatrixMultiplication
from automatic_diff_engine.tensor.functions.multiplication_function import MultiplicationFunction
from automatic_diff_engine.tensor.functions.subtraction_function import SubtractionFunction
from automatic_diff_engine.tensor.tensor_data import TensorData


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
                if type(node) is Tensor and node.requires_grad:
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

    def __add__(self, other: Self) -> Self:
        result = Tensor(
            data = AdditionFunction.forward(self.data, other.data),
            creator_func = AdditionFunction,
            creator_operands = [self, other]
        )

        return result

    def __sub__(self, other: Self) -> Self:
        result = Tensor(
            data=SubtractionFunction.forward(self.data, other.data),
            creator_func=SubtractionFunction,
            creator_operands=[self, other]
        )

        return result

    def __mul__(self, other: Self) -> Self:
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

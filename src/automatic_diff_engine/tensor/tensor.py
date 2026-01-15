from typing import Self, Optional

import numpy as np

from automatic_diff_engine.tensor.function import Function
from automatic_diff_engine.tensor.functions.addition_function import AdditionFunction
from automatic_diff_engine.tensor.functions.constant_function import ConstantFunction
from automatic_diff_engine.tensor.functions.exponential_function import ExponentialFunction
from automatic_diff_engine.tensor.functions.multiplication_function import MultiplicationFunction
from automatic_diff_engine.tensor.tensor_data import TensorData


class Tensor:
    def __init__(
            self,
            data: np.ndarray,
            creator_func: Function = ConstantFunction(),
            creator_operands = None,
            requires_grad = True,
    ):
        self.data = TensorData(np.array(data, dtype=np.float64), np.zeros_like(data, dtype=np.float64), requires_grad)
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
            grads = tensor.creator_func.backward(tensor.grad, *[t.data for t in tensor.creator_operands])
            if not isinstance(grads, tuple):
                grads = (grads,)

            for i in range(len(tensor.creator_operands)):
                if tensor.creator_operands[i].requires_grad:
                    tensor.creator_operands[i].grad += grads[i]

            # for i, grad in enumerate(grads):
            #     if i < len(tensor.creator_operands):
            #         creator_operand = tensor.creator_operands[i]
            #         if grad is not None and creator_operand.requires_grad:
            #             creator_operand.grad += grad

    def __repr__(self) -> str:
        return f"Tensor: {self.data}"

    def __add__(self, other: Self) -> Self:
        addition = AdditionFunction()
        result = Tensor(
            data = addition.forward(self.data, other.data),
            creator_func = addition,
            creator_operands = [self, other]
        )

        return result

    def __sub__(self, other: Self) -> Self:
        return Tensor(self.value - other.value)

    def __mul__(self, other: Self) -> Self:
        multiplication = MultiplicationFunction()
        result = Tensor(
            data = multiplication.forward(self.data, other.data),
            creator_func = multiplication,
            creator_operands = [self, other]
        )

        return result

    def __pow__(self, power, modulo=None):
        power = Tensor(
            data = power,
            creator_func = ConstantFunction(),
            creator_operands = [],
            requires_grad = False
        )
        exponential = ExponentialFunction()
        result = Tensor(
            data = exponential.forward(self.data, power),
            creator_func = exponential,
            creator_operands = [self, power]
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
        return np.shape(self.data)

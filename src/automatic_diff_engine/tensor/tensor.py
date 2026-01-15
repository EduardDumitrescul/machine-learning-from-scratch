from typing import Callable, Self, Optional

import numpy as np

from automatic_diff_engine.tensor.operation import Operation, Constant, AdditionFunction, MultiplicationFunction, ExponentialFunction


class Tensor:
    def __init__(
            self,
            data: Optional[np.ndarray],
            creator_func: Operation = Constant(),
            creator_operands = None,
            requires_grad = True,
    ):
        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.creator_func = creator_func
        self.creator_operands = creator_operands if creator_operands is not None else []
        self.requires_grad = requires_grad


    def shape(self) -> tuple:
        return np.shape(self.data)

    def backward(self):
        self.grad = np.ones(self.shape())

        topo: list[Self] = list()
        visited = set()
        def build_topo(node: Self) -> None:
            if node not in visited:
                visited.add(node)
                if type(node) is Tensor:
                    for next_node in node.creator_operands:
                        build_topo(next_node)
                    topo.append(node)

        build_topo(self)
        for tensor in reversed(topo):
            grads = tensor.creator_func.backward(tensor.grad)

            for i, grad in enumerate(grads):
                if i < len(tensor.creator_operands):
                    creator_operand = tensor.creator_operands[i]
                    if grad is not None and creator_operand.requires_grad:
                        creator_operand.grad += grad

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
        return Tensor(self.data - other.data)

    def __mul__(self, other: Self) -> Self:
        multiplication = MultiplicationFunction()
        result = Tensor(
            data = multiplication.forward(self.data, other.data),
            creator_func = multiplication,
            creator_operands = [self, other]
        )

        return result

    def __pow__(self, power, modulo=None):
        exponential = ExponentialFunction()
        result = Tensor(
            data = exponential.forward(self.data, power),
            creator_func = exponential,
            creator_operands = [self, power]
        )

        return result
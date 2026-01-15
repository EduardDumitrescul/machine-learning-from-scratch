from typing import Callable, Self, Optional

import numpy as np

class Tensor:
    def __init__(
            self,
            data: np.typing.NDArray,
            creators=None,
    ):
        self.data = np.array(data, dtype = np.float32)
        self.grad = np.zeros_like(self.data)
        self.creators: list[Self] = creators if creators is not None else list()
        self._backward_func: Callable[[], None] = lambda: None

    def shape(self) -> tuple:
        return np.shape(self.data)

    def backward(self):
        self.grad = np.ones(self.shape())

        topo: list[Self] = list()
        visited = set()
        def build_topo(node: Self) -> None:
            if node not in visited:
                visited.add(node)
                for next_node in node.creators:
                    build_topo(next_node)
                topo.append(node)

        build_topo(self)
        for tensor in reversed(topo):
            tensor._backward_func()
    
    def __repr__(self) -> str:
        return f"Tensor: {self.data}"

    def __add__(self, other: Self) -> Self:
        result = Tensor(
            data = self.data + other.data,
            creators = [self, other]
        )
        def _backward_func():
            self.grad += result.grad
            other.grad += result.grad

        result._backward_func = _backward_func
        return result

    def __sub__(self, other: Self) -> Self:
        return Tensor(self.data - other.data)

    def __mul__(self, other: Self) -> Self:
        result = Tensor(
            data = self.data * other.data,
            creators = [self, other]
        )

        def _backward_func():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward_func = _backward_func
        return result

    def __pow__(self, power, modulo=None):
        result = Tensor(
            data = self.data ** power,
            creators = [self]
        )

        def _backward_func():
            self.grad += power * self.data**(power-1) * result.grad

        result._backward_func = _backward_func
        return result
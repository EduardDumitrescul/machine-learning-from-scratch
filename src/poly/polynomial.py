from automatic_diff_engine.tensor import Tensor


def compute_polynomial(coefficients, x) -> Tensor:
    result = coefficients[0]
    for i in range(1, len(coefficients)):
        result = result * x + coefficients[i]
    return result
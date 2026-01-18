from automatic_diff_engine.tensor import Tensor


def binary_cross_entropy(labels: Tensor, probs: Tensor) -> Tensor:
    assert  labels.shape == probs.shape, f"Incompatible shapes: {labels.shape} and {probs.shape}"
    one = Tensor(1, requires_grad=False)
    zero = Tensor(0, requires_grad=False)
    loss = zero - (labels * probs.log() + (one - labels) * (one - probs).log())
    return loss.mean()
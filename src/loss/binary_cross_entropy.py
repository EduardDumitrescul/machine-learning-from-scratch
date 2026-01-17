from automatic_diff_engine.tensor import Tensor


def binary_cross_entropy(labels: Tensor, probs: Tensor) -> Tensor:
    one = Tensor(1, requires_grad=False)
    zero = Tensor(0, requires_grad=False)
    loss = zero - (labels * probs.log() + (one - labels) * (one - probs).log())
    return loss.mean()
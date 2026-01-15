from automatic_diff_engine.tensor.tensor import Tensor


def test_addition_backwards_scalar():
    a = Tensor(1)
    b = Tensor(4)

    result = a - b
    result.backward()

    assert result.grad.tolist() == [1]
    assert a.grad.tolist() == [1]
    assert b.grad.tolist() == [-1]

def test_addition_backwards_vector():
    a = Tensor([4, 5, 6])
    b = Tensor([1, 2, 3])

    result = a - b
    result.backward()

    assert result.grad.tolist() == [1, 1, 1]
    assert a.grad.tolist() == [1, 1, 1]
    assert b.grad.tolist() == [-1, -1, -1]
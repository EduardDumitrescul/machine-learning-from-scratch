import numpy as np

from loss.binary_cross_entropy import binary_cross_entropy
from src.automatic_diff_engine.tensor import Tensor


def test_bce_forward():
    # Test perfect match and partial match
    # labels: [1, 0], probs: [0.8, 0.2]
    labels = Tensor(np.array([1.0, 0.0]))
    probs = Tensor(np.array([0.8, 0.2]))

    loss = binary_cross_entropy(labels, probs)

    # Calculation: -1/2 * ( [1*ln(0.8) + 0*ln(0.2)] + [0*ln(0.2) + 1*ln(0.8)] )
    # = -1/2 * (ln(0.8) + ln(0.8)) = -ln(0.8) approx 0.22314
    expected = -np.log(0.8)
    assert np.allclose(loss.value, expected)


def test_bce_backward_basic():
    # Test gradient with respect to probabilities
    labels = Tensor(np.array([1.0]), requires_grad=False)
    probs = Tensor(np.array([0.5]), requires_grad=True)

    loss = binary_cross_entropy(labels, probs)
    loss.backward()

    # dL/dp = (1/N) * (p - y) / (p * (1 - p))
    # For N=1, y=1, p=0.5: (0.5 - 1) / (0.5 * 0.5) = -0.5 / 0.25 = -2.0
    assert np.allclose(probs.grad, [-2.0])


def test_bce_zero_loss():
    # Test loss when probabilities perfectly match labels (approaching limit)
    labels = Tensor(np.array([1.0, 0.0]))
    # Using values very close to 1 and 0
    probs = Tensor(np.array([0.9999, 0.0001]), requires_grad=True)

    loss = binary_cross_entropy(labels, probs)

    assert loss.value < 1e-3


def test_bce_multi_element_grad():
    labels = Tensor(np.array([1.0, 0.0]), requires_grad=False)
    probs = Tensor(np.array([0.4, 0.6]), requires_grad=True)

    loss = binary_cross_entropy(labels, probs)
    loss.backward()

    # N=2
    # p1: 0.5 * (0.4 - 1) / (0.4 * 0.6) = 0.5 * (-0.6 / 0.24) = -1.25
    # p2: 0.5 * (0.6 - 0) / (0.6 * 0.4) = 0.5 * (0.6 / 0.24) = 1.25
    expected_grad = np.array([-1.25, 1.25])
    assert np.allclose(probs.grad, expected_grad)
"""``_quantile`` must be bit-identical to ``np.quantile``'s linear method."""

import numpy as np
import pytest

from bayesianbandits.policies._upper_confidence_bound import _quantile


@pytest.mark.parametrize("n_draws", [1, 2, 8, 100])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.68, 0.95, 1.0])
def test_matches_numpy(n_draws: int, q: float) -> None:
    samples = np.random.default_rng(0).normal(size=(3, 5, n_draws))
    samples[0, 0] = 1.0  # ties

    assert np.array_equal(_quantile(samples, q), np.quantile(samples, q, axis=-1))

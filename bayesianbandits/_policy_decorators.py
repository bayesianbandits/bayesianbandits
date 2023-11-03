from __future__ import annotations

from typing import Callable, Dict, List, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._arm import Arm

ArmChoicePolicy = Callable[
    [Dict[str, Arm], NDArray[np.float_], np.random.Generator],
    Union[Arm, List[Arm]],
]


def epsilon_greedy(
    epsilon: float = 0.1,
    *,
    samples: int = 1000,
) -> ArmChoicePolicy:
    """Creates an epsilon-greedy choice algorithm. To be used with the
    `Bandit` class.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of choosing a random arm.
    samples : int, default=1000
        Number of samples to use to compute the mean of the posterior.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using epsilon-greedy.
    """

    def _choose_arm(
        arms: Dict[str, Arm],
        X: NDArray[np.float_],
        rng: np.random.Generator,
    ) -> Union[Arm, List[Arm]]:
        """Choose an arm using epsilon-greedy."""

        arm_list = list(arms.values())

        means = np.stack(
            tuple(_compute_arm_mean(arm, X, samples=samples) for arm in arm_list)
        )

        choices = _return_based_on_size(arm_list, means)

        if not isinstance(choices, list):
            if rng.random() < epsilon:
                return rng.choice(arm_list)  # type: ignore
            else:
                return choices

        choice_idx_to_explore = rng.random(size=len(choices)) < epsilon

        final_choices: List[Arm] = []
        for explore, choice in zip(choice_idx_to_explore, choices):
            if explore:
                final_choices.append(rng.choice(arm_list))  # type: ignore
            else:
                final_choices.append(choice)

        return final_choices

    return _choose_arm


def upper_confidence_bound(
    alpha: float = 0.68,
    *,
    samples: int = 1000,
) -> ArmChoicePolicy:
    """Creates a UCB choice algorithm. To be used with the
    `Bandit` class.

    Actually uses the upper bound of the one-sided credible interval,
    which will deviate from the upper bound of the one-sided confidence
    interval depending on the strength of the prior.

    Parameters
    ----------
    alpha : float, default=0.68
        Upper bound of the one-sided prediction interval.
    samples : int, default=1000
        Number of samples to use to compute the upper bound of the
        credible interval. The larger `alpha` is, the larger `samples`
        should be.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using UCB.

    Notes
    -----
    A deeper analysis of this Bayesian UCB algorithm can be found in
    [1].

    References
    ----------
    [1] E. Kaufmann, O. Cappé, and A. Garivier, “On Bayesian upper confidence
        bounds for bandit problems,” In Proceedings of the 15th International
        Conference on Artificial Intelligence and Statistics, 2012.

    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    def _choose_arm(
        arms: Dict[str, Arm],
        X: NDArray[np.float_],
        rng: np.random.Generator,
    ) -> Union[Arm, List[Arm]]:
        """Choose an arm using UCB1."""

        arm_list = list(arms.values())

        # Compute the upper bound of the one-sided credible interval for
        # each arm.
        upper_bounds = np.stack(
            tuple(
                _compute_arm_upper_bound(arm, X, alpha=alpha, samples=samples)
                for arm in arm_list
            )
        )

        # Return the arm(s) with the largest upper bound.
        return _return_based_on_size(arm_list, upper_bounds)

    return _choose_arm


def thompson_sampling() -> ArmChoicePolicy:
    """Creates a Thompson sampling choice algorithm. To be used with the
    `Bandit` class.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using Thompson sampling.

    Notes
    -----
    This is a very simple implementation of Thompson sampling, identical
    to the one used in [1]. This generally works well in practice and
    has no hyperparameters to tune.

    References
    ----------
    [1] D. Russo, B. Van Row, A. Kazerouni, I. Osband, and Z. Wen, “A
        Tutorial on Thompson Sampling,” Foundations and Trends® in Machine
        Learning, vol. 11, no. 1, pp. 1-96, 2018.
    """

    def _choose_arm(
        arms: Dict[str, Arm],
        X: NDArray[np.float_],
        rng: np.random.Generator,
    ) -> Union[Arm, List[Arm]]:
        """Choose an arm using Thompson sampling."""

        arm_list = list(arms.values())

        # Sample from the posterior distribution for each arm.
        posterior_summaries = np.stack(
            tuple(_draw_one_sample(arm, X) for arm in arm_list)
        )

        return _return_based_on_size(arm_list, posterior_summaries)

    return _choose_arm


def _return_based_on_size(
    arm_list: List[Arm],
    posterior_summaries: NDArray[np.float_],
):
    best_arm_indexes = cast(
        NDArray[np.int_], np.atleast_1d(np.argmax(posterior_summaries, axis=0))
    )

    if len(best_arm_indexes) == 1:
        return arm_list[best_arm_indexes.item()]

    else:
        return [arm_list[cast(int, i)] for i in best_arm_indexes]


def _draw_one_sample(arm: Arm, X: NDArray[np.float_]) -> NDArray[np.float_]:
    """Draw one sample from the posterior distribution for the arm."""
    return arm.sample(X, size=1).squeeze(axis=0)


def _compute_arm_upper_bound(
    arm: Arm,
    X: NDArray[np.float_],
    *,
    alpha: float = 0.68,
    samples: int = 1000,
) -> np.float_:
    """Compute the upper bound of a one-sided credible interval with size
    `alpha` from the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)

    return np.quantile(posterior_samples, q=alpha, axis=0)  # type: ignore


def _compute_arm_mean(
    arm: Arm,
    X: NDArray[np.float_],
    *,
    samples: int = 1000,
) -> np.float_:
    """Compute the mean of the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)
    return np.mean(posterior_samples, axis=0)

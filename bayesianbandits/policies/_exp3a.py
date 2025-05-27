"""
EXP3.A: An average-based, anytime variant of EXP3 for adversarial and non-stationary bandits.

This module implements EXP3.A, which maintains the adversarial robustness of EXP3
while using average rewards instead of cumulative sums, enabling better adaptivity
and numerical stability.
"""

from typing import TYPE_CHECKING, List, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array

from bayesianbandits._typing import DecayingLearner

if TYPE_CHECKING:
    from bayesianbandits import Arm

LT = TypeVar("LT", bound=DecayingLearner)
T = TypeVar("T")


class EXP3A:
    """
    EXP3.A: Average-based anytime variant of EXP3 for adversarial bandits.

    This algorithm provides adversarial robustness while solving several practical
    limitations of traditional adversarial bandit algorithms like EXP3/EXP4.

    Core adversarial mechanism:

    - Importance weighting: Uses weight = 1/P(arm) to debias reward estimates
    - Forced exploration: γ-mixing guarantees P(arm) ≥ γ/K for all arms
    - No assumptions: Works with arbitrary reward sequences, including adversarial

    Key practical advantages:

    1. **Contextual by default**: Handles contexts naturally through Bayesian
       learners without the complexity of explicit policy enumeration (EXP4)

    2. **Anytime algorithm**: No need to know time horizon T in advance or tune
       η based on expected runtime - just set η based on reward scale

    3. **Adaptive to change**: Tracks moving targets and changing adversaries
       by using this library's Bayesian learners, which support both sample
       weights and Bayesian forgetting via variance-increasing decay

    4. **Numerical stability**: Weights stay bounded by exp(η * max_reward),
       avoiding the numerical overflow issues of cumulative approaches

    When to use this algorithm:

    - Adversarial or non-stationary environments
    - Unknown or variable time horizons
    - Contextual problems where adversary can adapt based on context
    - Long-running experiments where numerical stability matters
    - When you need to detect and adapt to strategy changes

    The importance weighting ensures each learner estimates the average reward
    it would receive under uniform sampling, preventing adversaries from
    exploiting the algorithm's selection bias.

    Parameters
    ----------
    gamma : float, default=0.1
        Exploration rate. Each arm pulled with probability at least γ/K.
        Higher values force more exploration.
    eta : float, default=1.0
        Temperature for exponential weights. Higher values create sharper
        distinctions between arms. Unlike EXP3, doesn't need scaling with
        horizon since we use averages not sums.
    samples : int, default=100
        Number of samples to use for computing expected rewards via Monte Carlo.

    Examples
    --------
    >>> from bayesianbandits import Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import ContextualAgent
    >>>
    >>> # Create arms with Bayesian learners
    >>> arms = [
    ...     Arm(action_token=f"action_{i}",
    ...         learner=NormalInverseGammaRegressor())
    ...     for i in range(3)
    ... ]
    >>>
    >>> # Initialize adversarial policy
    >>> policy = EXP3A(gamma=0.1, eta=2.0)
    >>>
    >>> # Create agent
    >>> agent = ContextualAgent(arms, policy)
    >>>
    >>> # Pull arms and update with importance-weighted rewards
    >>> X = np.array([[1.0, 2.0]])
    >>> action = agent.pull(X)
    >>> reward = np.array([0.5])
    >>> agent.update(X, reward)

    Notes
    -----
    This algorithm uses the same Bayesian learners as the rest of the library
    for tracking reward estimates. The "average-based" refers to how rewards
    are aggregated into a posterior, not to the underlying estimation method.

    The algorithm excels in practical scenarios such as:

    - A/B testing where user preferences shift over time
    - Recommendation systems facing adversarial users or competitors
    - Online advertising with strategic bidders
    - Resource allocation under attack or manipulation
    - Any setting where both exploration and robustness are critical

    Based on the importance weighting technique from EXP3 (Auer et al., 2002),
    but uses average rewards instead of cumulative sums for better adaptivity.
    """

    def __init__(self, gamma: float = 0.1, eta: float = 1.0, samples: int = 100):
        self.gamma = gamma
        self.eta = eta
        self.samples = samples

    def __repr__(self) -> str:
        return f"EXP3A(gamma={self.gamma}, eta={self.eta}, samples={self.samples})"

    def __call__(
        self,
        arms: List["Arm[LT, T]"],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List["Arm[LT, T]"]:
        """
        Select arms according to exponential weights with forced exploration.

        Parameters
        ----------
        arms : List[Arm]
            Available arms to choose from
        X : array-like of shape (n_samples, n_features)
            Context matrix
        rng : np.random.Generator
            Random number generator for sampling

        Returns
        -------
        List[Arm]
            Selected arms, one per context
        """
        assert X.shape is not None, "Context matrix X must not be empty"

        # Get expected rewards via Monte Carlo estimation
        # This works for all arm types and uses reward_function automatically
        rewards = np.stack(
            [arm.sample(X, size=self.samples).mean(axis=0) for arm in arms]
        )

        # Exponential weights with numerical stability
        weights = np.exp(self.eta * (rewards - rewards.max(axis=0)))

        # Mix with uniform exploration - the classic EXP3 probability formula
        probs = (1 - self.gamma) * weights / weights.sum(axis=0)
        probs += self.gamma / len(arms)

        # Sample arms according to probabilities
        choices = [rng.choice(len(arms), p=probs[:, i]) for i in range(X.shape[0])]

        return [arms[i] for i in choices]

    def update(
        self,
        arm: "Arm[LT, T]",
        X: Union[NDArray[np.float64], csc_array],
        y: NDArray[np.float64],
        all_arms: List["Arm[LT, T]"],
        rng: np.random.Generator,
    ) -> None:
        """
        Update arm with importance-weighted reward.

        The importance weighting ensures unbiased estimates under adversarial
        reward generation, making the algorithm robust to arbitrary adversaries.

        Parameters
        ----------
        arm : Arm
            The arm that was pulled
        X : array-like of shape (n_samples, n_features)
            Context matrix
        y : array-like of shape (n_samples,)
            Observed rewards
        all_arms : List[Arm]
            All available arms (needed to recompute probabilities)
        rng : np.random.Generator
            Random number generator (unused but part of interface)
        """
        # Recompute probabilities (stateless design)
        rewards = np.stack(
            [a.sample(X, size=self.samples).mean(axis=0) for a in all_arms]
        )
        weights = np.exp(self.eta * (rewards - rewards.max(axis=0)))
        probs = (1 - self.gamma) * weights / weights.sum(axis=0)
        probs += self.gamma / len(all_arms)

        # Importance weighting: the key to adversarial robustness
        # This ensures each learner estimates E[reward | uniform sampling]
        arm_idx = all_arms.index(arm)
        importance_weights = 1.0 / probs[arm_idx]

        arm.update(X, y, sample_weight=importance_weights)

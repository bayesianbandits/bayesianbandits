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

    - Importance weighting: Uses weight = 1/(P(arm) + ix_gamma) to debias reward estimates
    - Optional forced exploration: γ-mixing guarantees P(arm) ≥ γ/K when gamma > 0
    - No assumptions: Works with arbitrary reward sequences, including adversarial

    Algorithm variants:

    1. **EXP3-IX** (Neu, 2015) [2]_ - Default:
       - Set `gamma = 0` and `ix_gamma > 0` (default: eta/2)
       - No forced exploration (pure exponential weights)
       - Regularized importance weights: 1/(P(arm) + γ_ix)
       - Better empirical performance and high-probability bounds

    2. **Standard EXP3** (Auer et al., 2002) [1]_:
       - Set `gamma > 0` and `ix_gamma = 0`
       - Uses forced exploration via γ-mixing
       - Unbiased importance weights: 1/P(arm)

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
    gamma : float, default=0.0
        Exploration rate for forced exploration. Each arm pulled with probability
        at least γ/K when gamma > 0. Default is 0 (no forced exploration).
    eta : float, default=1.0
        Temperature for exponential weights. Higher values create sharper
        distinctions between arms. Unlike standard EXP3, doesn't need scaling
        with horizon since we use averages not sums.

        Note: The appropriate value of eta depends on the scale of your rewards.
        Larger reward scales require smaller eta values to avoid numerical issues
        and maintain meaningful exploration.
    ix_gamma : float or None, default=None
        Regularization parameter for importance weights. If None, defaults to
        eta/2 as recommended by Neu (2015). Use 0 for unbiased weights. IX stands for
        Implicit eXploration.

        Note: This is unfortunately also called "gamma" in Neu (2015),
        but it serves a different purpose than the exploration rate gamma. It does
        serve to encourage exploration, but by regularizing the importance
        weights (smaller updates, spend more time close to the prior for each arm) rather
        than by mixing exploration into the arm selection probabilities.
    samples : int, default=100
        Number of samples to use for computing expected rewards via Monte Carlo. Higher
        values improve accuracy but increase computational cost. The more parameters
        your Bayesian learners have, the more samples you may need to get stable estimates.


    References
    ----------
    .. [1] Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
       "The nonstochastic multiarmed bandit problem." SIAM Journal on Computing,
       32(1), 48-77.

    .. [2] Neu, G. (2015). "Explore no more: Improved high-probability regret
       bounds for non-stochastic bandits." Advances in Neural Information
       Processing Systems, 28.

    Examples
    --------
    EXP3-IX variant (default, recommended):

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
    >>> # Initialize EXP3-IX policy (default)
    >>> policy = EXP3A(eta=2.0)  # ix_gamma defaults to eta/2 = 1.0
    >>>
    >>> # Create agent
    >>> agent = ContextualAgent(arms, policy)

    Standard EXP3 with forced exploration:

    >>> # Initialize standard EXP3 policy
    >>> policy = EXP3A(gamma=0.1, eta=2.0, ix_gamma=0.0)
    >>>
    >>> # Create agent
    >>> agent = ContextualAgent(arms, policy)

    Notes
    -----
    This implementation is inspired by EXP3 (Auer et al., 2002) and EXP3-IX
    (Neu, 2015) but makes several practical modifications that may affect
    theoretical guarantees:

    - Uses average-based rewards instead of cumulative sums
    - Integrates with Bayesian learners rather than maintaining explicit weights
    - Employs Monte Carlo estimation for expected rewards
    - Supports variance decay for non-stationary environments

    While these modifications improve practical performance, the theoretical
    regret bounds from the original papers may not apply. This algorithm
    should be viewed as a practical variant that maintains the adversarial
    robustness intuition of EXP3 while adapting to real-world constraints.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        eta: float = 1.0,
        ix_gamma: Union[float, None] = None,
        samples: int = 100,
    ):
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if ix_gamma is not None and ix_gamma < 0:
            raise ValueError("ix_gamma must be non-negative")

        self.gamma = gamma
        self.eta = eta
        self.samples = samples
        # Default to eta/2 as recommended by Neu (2015)
        self.ix_gamma = eta / 2.0 if ix_gamma is None else ix_gamma

    def __repr__(self) -> str:
        return f"EXP3A(gamma={self.gamma}, eta={self.eta}, ix_gamma={self.ix_gamma}, samples={self.samples})"

    def __call__(
        self,
        arms: List["Arm[LT, T]"],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List["Arm[LT, T]"]:
        """
        Select arms according to exponential weights with optional exploration.

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
        rewards = np.stack(
            [arm.sample(X, size=self.samples).mean(axis=0) for arm in arms]
        )

        # Exponential weights with numerical stability
        weights = np.exp(self.eta * (rewards - rewards.max(axis=0)))

        # Compute probabilities (works for both variants)
        probs = (1 - self.gamma) * weights / weights.sum(axis=0) + self.gamma / len(
            arms
        )

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
        probs = (1 - self.gamma) * weights / weights.sum(axis=0) + self.gamma / len(
            all_arms
        )

        # Importance weighting: the key to adversarial robustness
        # This ensures each learner estimates E[reward | uniform sampling]
        arm_idx = all_arms.index(arm)
        importance_weights = 1.0 / (probs[arm_idx] + self.ix_gamma)

        arm.update(X, y, sample_weight=importance_weights)

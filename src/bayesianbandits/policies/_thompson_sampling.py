"""
Thompson sampling policy for Bayesian bandits.
"""

from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._argmax_probability import (
    argmax_probabilities,
    independent_argmax_probabilities,
)
from .._arm import (
    Arm,
    ContextType,
    TokenType,
    is_identity_function,
    resolve_predictive_moments,
)
from .._draw_kind import DrawKind
from ._base import PolicyDefaultUpdate


def _arm_moments(arm: Arm[ContextType, TokenType], X: ContextType) -> Any:
    """The arm's predictive moments, when they describe what ``select``
    compares: the learner's linear predictor under the identity reward."""
    if not is_identity_function(arm.reward_function):
        raise NotImplementedError(
            "ThompsonSampling's selection probabilities need the identity "
            "reward function: they are integrated from the posterior, which "
            "a reward function would have to be inverted through"
        )
    if type(arm).sample is not Arm.sample:
        raise NotImplementedError(
            f"{type(arm).__name__} overrides sample, so the learner's "
            "posterior does not describe its draws"
        )
    moments = resolve_predictive_moments(arm.learner)
    if moments is None:
        raise NotImplementedError(
            f"{type(arm.learner).__name__} has no predictive moments to "
            "integrate ThompsonSampling's selection probabilities from"
        )
    return moments(X)


class ThompsonSampling(PolicyDefaultUpdate[ContextType, TokenType]):
    """
    Policy object for Thompson sampling.

    Thompson sampling chooses the best arm with probability equal to the
    probability that the arm is the best arm. At each round, a single sample
    is drawn from each arm's posterior and the arm with the highest sample is
    selected:

    .. math::

        \\tilde{\\theta}_a \\sim p(\\theta_a \\mid \\mathcal{D}_a)
        \\quad \\forall a, \\qquad
        a^* = \\arg\\max_a \\; g_a(\\tilde{\\theta}_a)

    where :math:`g_a` is the reward function (possibly context-dependent) and
    :math:`\\mathcal{D}_a` is the data observed for arm :math:`a`.

    Examples
    --------
    >>> from bayesianbandits import Agent, Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import ThompsonSampling
    >>>
    >>> arms = [
    ...     Arm(f"arm_{i}", learner=NormalInverseGammaRegressor())
    ...     for i in range(3)
    ... ]
    >>> agent = Agent(arms, ThompsonSampling())

    See Also
    --------
    UpperConfidenceBound : Deterministic optimism via posterior quantiles.
        Explores more aggressively than Thompson sampling for uncertain arms.
    EpsilonGreedy : Simple exploration via random arm selection.
        Does not use posterior uncertainty to guide exploration.
    EXP3A : Adversarial robustness via importance-weighted updates.
        Use when reward generation may be adversarial or non-stochastic.

    Notes
    -----
    **Regret bounds (standard setting).** For the :math:`K`-armed stochastic
    bandit with Beta-Bernoulli or Gaussian conjugate models, Thompson sampling
    achieves a Bayesian expected regret of

    .. math::

        \\mathbb{E}[\\mathrm{Regret}(T)]
        = O\\!\\left(\\sqrt{KT \\ln K}\\right)

    and an asymptotically optimal problem-dependent bound of
    :math:`O\\!\\left(\\sum_{a:\\Delta_a>0}
    \\frac{\\ln T}{\\Delta_a}\\right)` matching the Lai-Robbins lower bound
    [2]_.

    **Applicability to this library.** The bounds above are proven for
    stationary, non-contextual bandits with exact conjugate posteriors [1]_.
    This library targets anytime, contextual, and potentially non-stationary
    problems, and the Bayesian learners may use approximate posteriors (e.g.
    Laplace approximations) or variance-increasing decay. Under these
    modifications the classical regret guarantees do not formally apply,
    though the core explore-exploit mechanism---probability matching via
    posterior sampling---is preserved.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing Systems
       24, 2249-2257.

    .. [2] Agrawal, S. and Goyal, N. (2012). "Analysis of Thompson sampling
       for the multi-armed bandit problem." Proceedings of the 25th Annual
       Conference on Learning Theory (COLT), JMLR W&CP 23, 39.1-39.26.
    """

    def __repr__(self) -> str:
        return "ThompsonSampling()"

    #: Requires joint draws (arms compared within a shared posterior
    #: draw), so marginal sampling must not be used. Draws one sample
    #: per pull, for which weight-space ``sample`` is always the
    #: cheaper joint path.
    consumes = DrawKind.JOINT

    @property
    def samples_needed(self) -> int:
        """Number of samples per arm per context needed for decision making."""
        return 1

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]:
        """Select arms based on pre-generated samples."""
        # samples shape: (n_arms, n_contexts, 1)
        values = samples[..., 0]  # Shape: (n_arms, n_contexts)

        if top_k is None:
            best_indices = values.argmax(axis=0)
            return [arms[idx] for idx in best_indices]
        else:
            # Use argpartition for efficiency, but handle case where top_k >= n_arms
            return [
                [
                    arms[idx]
                    for idx in np.argpartition(
                        -values[:, i], min(top_k, len(arms) - 1)
                    )[:top_k]
                ]
                for i in range(values.shape[1])
            ]

    def select_with_probabilities(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Not implemented: Thompson sampling's selection probability is the
        posterior probability that an arm's draw is the largest, which the
        single draw ``select`` consumes does not determine."""
        raise NotImplementedError(
            "ThompsonSampling's selection probabilities are the posterior "
            "probabilities that each arm's draw is the largest, which the "
            "single draw it selects from does not determine"
        )

    def call_with_probabilities(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Draw and select as ``__call__`` does, and return the posterior
        probability that each arm's draw is the largest: with one learner
        per arm the scores are independent, so it is a one-dimensional
        integral over each arm's predictive mean and variance."""
        moments = [_arm_moments(arm, X) for arm in arms]
        chosen = self.select(self._draw_samples(arms, X, 1), arms, rng)
        mean = np.stack([m.mean for m in moments], axis=1)
        var = np.stack([m.cov[:, 0, 0] for m in moments], axis=1)
        df = None
        if any(m.df is not None for m in moments):
            df = np.broadcast_to(
                [np.inf if m.df is None else m.df for m in moments], mean.shape
            )
        return chosen, independent_argmax_probabilities(mean, var, df)

    def probabilities_from_moments(
        self,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
        df: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """The posterior probability that each arm's draw is the largest
        for jointly distributed scores, ``mean`` ``(n, K)``, ``cov``
        ``(n, K, K)`` and ``df`` ``(n,)`` or None: an agent with a shared
        learner calls this in place of ``select_with_probabilities``.
        Independent scores take the exact one-dimensional integral."""
        off_diagonal = cov * (1.0 - np.eye(cov.shape[-1]))
        if not off_diagonal.any():
            var = np.diagonal(cov, axis1=1, axis2=2)
            return independent_argmax_probabilities(
                mean,
                var,
                None if df is None else np.broadcast_to(df[:, None], mean.shape),
            )
        return argmax_probabilities(mean, cov, df)

"""
Thompson sampling policy for Bayesian bandits.
"""

from typing import (
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType
from .._draw_kind import DrawKind
from ._base import ContextDraws, PolicyDefaultUpdate, _top_k_indices

# Cap on one chunk of propensity draws, (n_arms, n_contexts, chunk), in
# float64 elements: ~32 MB
_PROPENSITY_CHUNK_ELEMS = 2**22


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

    Parameters
    ----------
    propensity_samples : int, default=10_000
        Extra posterior draws ``pull_with_probabilities`` counts argmaxes
        over to estimate the selection probabilities, a positive integer.
        Selection itself always uses one draw. The draws come a few
        contexts at a time, except under a ``LipschitzContextualAgent``
        batch reward function, which may combine contexts within a draw:
        there every context is drawn together in chunks of at most
        ``2**22`` arm-context-draw cells, so once arms times contexts
        passes that each draw is a separate featurize-and-sample call.
        Pull such batches in smaller pieces.

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

    #: A class default too, so a policy pickled before the parameter
    #: existed still has it
    propensity_samples: int = 10_000

    def __repr__(self) -> str:
        return f"ThompsonSampling(propensity_samples={self.propensity_samples})"

    def __init__(self, propensity_samples: int = 10_000):
        if (
            isinstance(propensity_samples, bool)
            or not isinstance(propensity_samples, (int, np.integer))
            or propensity_samples < 1
        ):
            raise ValueError(
                "propensity_samples must be a positive integer, got "
                f"{propensity_samples!r}"
            )
        self.propensity_samples = int(propensity_samples)

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
        return [[arms[i] for i in row] for row in _top_k_indices(values, top_k)]

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

    def select_with_drawn_probabilities(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, 1)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        draw: ContextDraws,
        split_contexts: bool = True,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Select as ``select`` does, with the ``(n_contexts, n_arms)``
        fraction of draws each arm's is the largest in: the decision's own
        ``samples`` plus ``propensity_samples`` more from ``draw``.

        With ``split_contexts`` a few contexts take all their draws at a
        time, so a learner pays each context's setup once; otherwise every
        context is drawn together, in chunks of draws. Counting the
        decision's draw keeps the chosen arm's probability at least
        ``1 / (propensity_samples + 1)``, the resolution, so an IPS weight
        is at most ``propensity_samples + 1``."""
        if type(self).select is not ThompsonSampling.select:
            # the counts are the plain argmax, not what the override picks
            raise NotImplementedError(
                f"{type(self).__name__} overrides select, so counting argmaxes "
                "would not give the probabilities of its choices"
            )
        n_arms, n_contexts = samples.shape[:2]
        n_draws = self.propensity_samples
        wins = np.zeros((n_contexts, n_arms))
        wins[np.arange(n_contexts), samples[..., 0].argmax(axis=0)] = 1.0
        step = max(1, n_contexts)
        if split_contexts:
            step = max(1, _PROPENSITY_CHUNK_ELEMS // (n_arms * n_draws))
        for first in range(0, n_contexts, step):
            contexts = slice(first, min(first + step, n_contexts))
            width = contexts.stop - first
            offsets = np.arange(width)[:, None] * n_arms
            chunk = max(1, _PROPENSITY_CHUNK_ELEMS // (n_arms * width))
            for start in range(0, n_draws, chunk):
                picks = draw(contexts, min(chunk, n_draws - start)).argmax(axis=0)
                wins[contexts] += np.bincount(
                    (picks + offsets).ravel(), minlength=width * n_arms
                ).reshape(width, n_arms)
        probs = cast(NDArray[np.float64], wins / (n_draws + 1))
        return self.select(samples, arms, rng), probs

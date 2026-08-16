"""
Information-directed sampling policy for Bayesian bandits.
"""

from typing import (
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType
from .._draw_kind import DrawKind
from ._base import PolicyDefaultUpdate


def _vids_statistics(
    samples: NDArray[np.float64],
    alive: NDArray[np.bool_],
    masked: Optional[NDArray[np.float64]] = None,
    mu: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Monte Carlo estimates of expected regret and variance-based information gain.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Joint posterior samples, shape ``(n_arms, n_contexts, n_draws)``.
        Draw ``m`` must be coherent across arms within a context: the
        conditional means below condition on which arm is optimal *within
        the same draw*.
    alive : NDArray[np.bool_]
        Shape ``(n_arms, n_contexts)``. Dead arms are excluded from the
        argmax (they cannot be "the optimal arm"), and their returned
        statistics are meaningless.
    masked, mu : NDArray[np.float64], optional
        Precomputed ``np.where(alive[:, :, np.newaxis], samples, -np.inf)``
        and ``samples.mean(axis=-1)``. The top_k slot loop maintains these
        across slots rather than re-materializing them here each slot.

    Returns
    -------
    delta, v : NDArray[np.float64]
        Shape ``(n_arms, n_contexts)``. ``delta[a, c]`` estimates
        :math:`\\mathbb{E}[\\max_{a'} \\theta_{a'}] - \\mathbb{E}[\\theta_a]`
        over alive arms, clipped at zero. ``v[a, c]`` estimates
        :math:`\\sum_{a'} p(a') (\\mathbb{E}[\\theta_a \\mid A^* = a'] -
        \\mathbb{E}[\\theta_a])^2`, the variance of arm ``a``'s posterior
        mean under resolution of which alive arm is optimal.
    """
    n_arms, _, n_draws = samples.shape
    if masked is None:
        masked = np.where(alive[:, :, np.newaxis], samples, -np.inf)
    means: NDArray[np.float64] = (
        samples.mean(axis=-1) if mu is None else mu
    )  # (n_arms, n_contexts)
    argmax_idx = masked.argmax(axis=0)  # (n_contexts, n_draws)
    # the values at the argmax are the max: gather them rather than
    # sweeping the full (n_arms, n_contexts, n_draws) tensor a second time
    rho_star = np.take_along_axis(masked, argmax_idx[np.newaxis], axis=0)[0].mean(
        axis=-1
    )  # (n_contexts,)
    delta = np.clip(rho_star[np.newaxis, :] - means, 0.0, None)

    onehot = (argmax_idx[:, :, np.newaxis] == np.arange(n_arms)).astype(np.float64)
    counts = onehot.sum(axis=1).T  # (n_arms, n_contexts); zero for dead arms
    p_opt = counts / n_draws
    # cond_sums[a, b, c]: sum of arm a's draws over the draws where b is
    # optimal, as a batched GEMM over contexts (much faster than einsum here)
    cond_sums = np.matmul(samples.transpose(1, 0, 2), onehot).transpose(1, 2, 0)
    mu_cond = cond_sums / np.maximum(counts, 1.0)[np.newaxis, :, :]
    # counts == 0 leaves mu_cond at 0, but p_opt == 0 zeroes the term anyway
    v = np.einsum("bc,abc->ac", p_opt, (mu_cond - means[:, np.newaxis, :]) ** 2)
    return delta, v


def _optimal_two_point(
    delta: NDArray[np.float64],
    v: NDArray[np.float64],
    alive: NDArray[np.bool_],
) -> Tuple[
    NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]
]:
    """Per context, the two-point distribution minimizing the information ratio.

    The minimizer of :math:`\\Delta(\\pi)^2 / v(\\pi)` over distributions
    :math:`\\pi` is supported on at most two arms (Russo & Van Roy 2018,
    Prop. 6), so it suffices to scan all ordered pairs ``(a, b)``. For a
    fixed pair, with mixing weight ``q`` on ``a``, the ratio is
    ``(A q + B)^2 / (C q + D)`` where ``A = delta_a - delta_b``,
    ``B = delta_b``, ``C = v_a - v_b``, ``D = v_b``; its stationary points
    are ``q = -B / A`` (zero regret) and ``q = (C B - 2 A D) / (A C)``, so
    the minimum over ``q`` in ``[0, 1]`` is attained at one of those
    (clipped) or an endpoint.

    Ratio conventions: a mixture with exactly zero regret has ratio 0
    (optimal no matter the gain); positive regret with zero gain has ratio
    ``inf``. Dead arms are excluded from the scan.

    Returns
    -------
    a_idx, b_idx, q, ratio : NDArray
        Each shape ``(n_contexts,)``: play ``a_idx`` with probability ``q``,
        else ``b_idx``, achieving information ratio ``ratio``. An
        infinite ``ratio`` means no alive pair has any information gain
        with nonzero regret everywhere (the posterior has resolved).
    """
    n_arms, n_contexts = delta.shape
    a_delta = delta[:, np.newaxis, :] - delta[np.newaxis, :, :]  # (K, K, C)
    b_delta = np.broadcast_to(delta[np.newaxis, :, :], a_delta.shape)
    c_gain = v[:, np.newaxis, :] - v[np.newaxis, :, :]
    d_gain = np.broadcast_to(v[np.newaxis, :, :], a_delta.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        root_zero = -b_delta / a_delta
        root_interior = (c_gain * b_delta - 2.0 * a_delta * d_gain) / (a_delta * c_gain)

    # Running elementwise minimum over the four q candidates, rather than
    # stacking them into (4, K, K, C) tensors: peak memory stays a few
    # (K, K, C) arrays instead of several 4x-sized ones
    best_ratio = np.full(a_delta.shape, np.inf)
    best_q = np.zeros(a_delta.shape)
    q_candidates = (
        0.0,
        1.0,
        np.clip(np.nan_to_num(root_zero, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
        np.clip(
            np.nan_to_num(root_interior, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        ),
    )
    for q_cand in q_candidates:
        num = (a_delta * q_cand + b_delta) ** 2
        den = c_gain * q_cand + d_gain
        ratio = np.full_like(num, np.inf)
        np.divide(num, den, out=ratio, where=den > 0.0)
        ratio[num == 0.0] = 0.0
        better = ratio < best_ratio
        best_q = np.where(better, q_cand, best_q)
        np.minimum(best_ratio, ratio, out=best_ratio)

    pair_alive = alive[:, np.newaxis, :] & alive[np.newaxis, :, :]
    best_ratio = np.where(pair_alive, best_ratio, np.inf)

    flat_ratio = best_ratio.reshape(n_arms * n_arms, n_contexts)
    pair_idx = flat_ratio.argmin(axis=0)  # (C,)
    ctx_idx = np.arange(n_contexts)
    a_idx = pair_idx // n_arms
    b_idx = pair_idx % n_arms
    q = best_q.reshape(n_arms * n_arms, n_contexts)[pair_idx, ctx_idx]
    return a_idx, b_idx, q, flat_ratio[pair_idx, ctx_idx]


def _sample_information_ratio_minimizer(
    delta: NDArray[np.float64],
    v: NDArray[np.float64],
    alive: NDArray[np.bool_],
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Per context, sample an arm from the information-ratio-minimizing distribution.

    Falls back to the greedy arm (``argmin delta`` among alive arms) in
    contexts where every pair's ratio is infinite: the posterior has
    resolved and there is no information left to buy.

    Returns
    -------
    NDArray[np.intp]
        Shape ``(n_contexts,)``, the sampled arm index per context.
    """
    a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
    picks = np.where(rng.random(delta.shape[1]) < q, a_idx, b_idx)

    resolved = ~np.isfinite(ratio)
    if np.any(resolved):
        greedy = np.where(alive, delta, np.inf).argmin(axis=0)
        picks = np.where(resolved, greedy, picks)
    return picks


class InformationDirectedSampling(PolicyDefaultUpdate[ContextType, TokenType]):
    """
    Policy object for variance-based information-directed sampling (IDS).

    At each round, IDS plays the (possibly randomized) action distribution
    minimizing the information ratio, the squared expected regret per unit
    of information gained about the identity of the optimal arm:

    .. math::

        \\pi^* = \\arg\\min_{\\pi \\in \\Delta_K} \\;
        \\frac{\\left(\\sum_a \\pi_a \\Delta_a\\right)^2}
             {\\sum_a \\pi_a v_a}

    where both terms are estimated from ``samples`` joint Monte Carlo draws
    :math:`\\theta^{(m)}` of the arms' rewards:

    .. math::

        \\Delta_a = \\mathbb{E}\\left[\\max_{a'} \\theta_{a'}\\right]
        - \\mathbb{E}[\\theta_a],
        \\qquad
        v_a = \\sum_{a'} p(A^* = a')
        \\left(\\mathbb{E}[\\theta_a \\mid A^* = a'] -
        \\mathbb{E}[\\theta_a]\\right)^2

    :math:`v_a` is the variance-based information gain of Russo & Van Roy
    [1]_: how much arm :math:`a`'s posterior mean moves upon learning which
    arm is optimal. It is a lower bound on the mutual information
    :math:`I(A^*; Y_a)` (up to a common noise factor) and is what makes IDS
    exploit cross-arm correlation: with a shared learner, observing one arm
    can be highly informative about another arm being optimal. The
    minimizing distribution is supported on at most two arms, found here by
    a closed-form scan over all pairs.

    Parameters
    ----------
    samples : int, default=1000
        Number of joint posterior samples used to estimate the regret and
        information-gain terms.

    Examples
    --------
    >>> from bayesianbandits import Agent, Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import InformationDirectedSampling
    >>>
    >>> arms = [
    ...     Arm(f"arm_{i}", learner=NormalInverseGammaRegressor())
    ...     for i in range(3)
    ... ]
    >>> agent = Agent(arms, InformationDirectedSampling())

    Fewer Monte Carlo draws trade estimation accuracy for speed:

    >>> agent = Agent(arms, InformationDirectedSampling(samples=500))

    See Also
    --------
    ThompsonSampling : Randomized exploration via posterior sampling.
        Cheaper per decision; explores proportionally to uncertainty but
        does not direct exploration toward informative arms.
    UpperConfidenceBound : Deterministic optimism via posterior quantiles.
    EpsilonGreedy : Simple exploration via random arm selection.

    Notes
    -----
    **Regret bounds (standard setting).** Russo & Van Roy [1]_ show that
    variance-based IDS satisfies the same
    :math:`\\mathbb{E}[\\mathrm{Regret}(T)] \\le \\sqrt{K T \\log K / 2}`
    Bayesian bound as the information-ratio analysis of Thompson sampling,
    while its per-round ratio minimization can be substantially smaller in
    structured problems (correlated arms, shared parameters), where
    Thompson sampling may repeatedly re-explore arms whose value is already
    implied by other observations.

    **Joint draws matter.** The conditional means inside :math:`v_a`
    condition on which arm is optimal *within the same posterior draw*, so
    this policy requires coherent joint samples across arms
    (``consumes = DrawKind.CONTEXT_JOINT``). With a shared learner
    (:class:`~bayesianbandits.LipschitzContextualAgent` or a shared
    pipeline), joint weight-space draws provide this. When arms have
    independent learners, independent per-arm draws *are* the correct joint
    distribution. If a shared learner cannot be batch-sampled, the per-arm
    fallback draws are incoherent across arms and :math:`v_a` degrades to
    its independent-arms form: still a valid algorithm, but blind to
    cross-arm correlation.

    **top_k semantics.** ``top_k=k`` returns k sequential IDS draws without
    replacement: slot 1 is exact IDS, and each later slot re-estimates
    :math:`\\Delta` and :math:`v` restricted to the remaining arms (the
    subgame with chosen arms removed), then plays exact IDS on it. Slot
    order is meaningful and ``top_k=1`` matches the base policy in
    distribution. No formal slate-bandit guarantee is claimed for
    :math:`k > 1`.

    **Applicability to this library.** The bounds above assume stationary
    rewards and exact posteriors, and the observation-noise factor relating
    :math:`v_a` to mutual information is common across arms only when
    arms share a likelihood (e.g. one shared learner, or homoscedastic
    Gaussian arms). Contextual features, approximate posteriors, and
    variance-increasing decay fall outside the formal analysis, though the
    regret-per-information principle driving exploration is preserved.

    References
    ----------
    .. [1] Russo, D. and Van Roy, B. (2018). "Learning to optimize via
       information-directed sampling." Operations Research 66(1), 230-252.

    .. [2] Russo, D. and Van Roy, B. (2016). "An information-theoretic
       analysis of Thompson sampling." Journal of Machine Learning Research
       17(68), 1-30.
    """

    def __repr__(self) -> str:
        return f"InformationDirectedSampling(samples={self.samples})"

    def __init__(self, samples: int = 1000):
        self.samples = samples

    #: Information gain conditions on which arm is the argmax within a
    #: draw, so arms must be jointly distributed; decisions are solved
    #: per context, so the dependence across contexts is never read.
    #: That is exactly ``CONTEXT_JOINT``, which lets an agent serve this
    #: with per-context reward-space blocks -- far cheaper than fully
    #: joint draws at this policy's sample counts.
    consumes = DrawKind.CONTEXT_JOINT

    @property
    def samples_needed(self) -> int:
        """Number of samples per arm per context needed for decision making."""
        return self.samples

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
        """Select arms by minimizing the information ratio over pre-generated samples."""
        n_arms, n_contexts, _ = samples.shape
        n_slots = 1 if top_k is None else min(top_k, n_arms)

        alive = np.ones((n_arms, n_contexts), dtype=np.bool_)
        # mu is alive-invariant; masked is maintained in place across
        # slots (dead arms' draws set to -inf) instead of re-materialized
        mu = samples.mean(axis=-1)
        masked = samples if n_slots == 1 else samples.astype(np.float64, copy=True)
        ctx_idx = np.arange(n_contexts)
        choices = np.empty((n_slots, n_contexts), dtype=np.intp)
        for slot in range(n_slots):
            delta, v = _vids_statistics(samples, alive, masked=masked, mu=mu)
            picks = _sample_information_ratio_minimizer(delta, v, alive, rng)
            choices[slot] = picks
            alive[picks, ctx_idx] = False
            if slot + 1 < n_slots:
                masked[picks, ctx_idx, :] = -np.inf

        if top_k is None:
            return [arms[idx] for idx in choices[0]]
        return [
            [arms[choices[slot, ctx]] for slot in range(n_slots)]
            for ctx in range(n_contexts)
        ]

    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]:
        """Choose arm(s) using information-directed sampling."""
        samples = self._draw_samples(arms, X, self.samples_needed)
        return self.select(samples, arms, rng, top_k)

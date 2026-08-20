"""
Information-directed sampling policy for Bayesian bandits.
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
from .._blas_helpers import draw_contiguous
from .._draw_kind import DrawKind
from ._base import PolicyDefaultUpdate


def _finite_draws(
    samples: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Replace non-finite draws so one overflow cannot pin a decision.

    A single ``inf`` or ``nan`` (a log-link GLM overflowing, say) makes
    that context's regret and gain non-finite for *every* arm, and the
    resolved-posterior fallback would then deterministically play the
    overflowed arm on every call. Each offending draw is replaced, for
    all arms of its context, by a random finite draw of the same
    context, keeping within-draw coherence; a context with no finite
    draw is zeroed, leaving its arms tied. Finite inputs pass through.
    """
    finite = np.isfinite(samples)
    if finite.all():
        return samples
    samples = samples.copy()
    good_draw = cast(NDArray[np.bool_], np.all(finite, axis=0))  # (n_contexts, n_draws)
    for ctx in np.flatnonzero(~np.all(good_draw, axis=1)):
        good = np.flatnonzero(good_draw[ctx])
        bad = np.flatnonzero(~good_draw[ctx])
        if good.size == 0:
            samples[:, ctx, :] = 0.0
        else:
            samples[:, ctx, bad] = samples[:, ctx, rng.choice(good, size=bad.size)]
    return samples


def _vids_statistics(
    samples: NDArray[np.float64],
    alive: NDArray[np.bool_],
    masked: Optional[NDArray[np.float64]] = None,
    indicator: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Monte Carlo regret and information-gain estimates, ``(n_arms, n_contexts)``.

    ``delta[a, c]`` estimates :math:`\\mathbb{E}[\\max_{a'} \\theta_{a'}] -
    \\mathbb{E}[\\theta_a]` over alive arms, clipped at zero; ``v[a, c]``
    estimates :math:`\\sum_{a'} p(a') (\\mathbb{E}[\\theta_a \\mid A^* = a']
    - \\mathbb{E}[\\theta_a])^2`, how much arm ``a``'s mean moves upon
    learning which alive arm is optimal. ``samples`` is
    ``(n_arms, n_contexts, n_draws)`` and must be jointly coherent across
    arms within a draw; dead arms are excluded from the argmax and their
    returned statistics are meaningless. ``masked`` and ``indicator``
    let the top_k slot loop reuse the alive-masked tensor and the
    boolean scratch across slots. Every entry of ``samples`` must be
    finite: one ``inf`` or ``nan`` draw poisons its context's row sums.

    Only arms winning at least one draw enter the conditioning
    (:math:`p(A^* = b) = 0` zeroes every other term exactly), so the
    conditional-sum GEMM runs over the ``M`` ever-optimal arms per
    context rather than all ``K``: ``O(K M S)`` with ``M`` typically
    single digits.
    """
    n_arms, n_contexts, n_draws = samples.shape
    if masked is None:
        masked = np.where(alive[:, :, np.newaxis], samples, -np.inf)

    # max + equality pass: cheaper than argmax over the leading axis,
    # and downstream wants the indicator, not the label
    rho = masked.max(axis=0)  # (n_contexts, n_draws)
    is_opt: NDArray[np.bool_] = (
        np.empty_like(samples, dtype=np.bool_) if indicator is None else indicator
    )
    np.equal(masked, rho, out=is_opt)
    # a uint16 accumulator beats bool->int64 promotion ~4x; exact below 2**16
    count_dtype = np.uint16 if n_draws < 2**16 else np.int64
    counts = np.add.reduce(
        is_opt.view(np.uint8), axis=-1, dtype=count_dtype
    )  # (n_arms, n_contexts); zero for dead arms

    # gather the ever-optimal arms per context; padding with
    # never-optimal arms contributes zero weight
    ever = counts > 0
    m_max = int(ever.sum(axis=0).max())
    opt_idx = np.argsort(~ever, axis=0, kind="stable")[:m_max].T  # (C, M)
    ctx_col = np.arange(n_contexts)[:, np.newaxis]
    weights = is_opt[opt_idx, ctx_col, :].astype(np.float64)  # (C, M, n_draws)
    if int(counts.sum()) != n_contexts * n_draws:
        # exact ties (e.g. duplicate arms under a shared learner): split
        # a tied draw evenly so p_opt stays a probability
        weights /= weights.sum(axis=1, keepdims=True)
    counts_sub = weights.sum(axis=-1)  # (C, M)
    p_opt = counts_sub / n_draws
    # cond_sums[c, a, b]: arm a's draw total where ever-optimal arm b wins
    cond_sums = np.matmul(samples.transpose(1, 0, 2), weights.transpose(0, 2, 1))
    row_sums = cond_sums.sum(axis=-1)  # (C, K): each arm's full draw sum
    # the weight columns partition the draws, so the row sums are
    # already each arm's draw total: no extra pass for the means
    means = row_sums.T / n_draws  # (n_arms, n_contexts)

    # E[max] from the same GEMM: the winning arm's entry sums rho over
    # its winning draws (ties still add rho exactly once). Regret as a
    # difference of cond_sums entries keeps an always-optimal arm's
    # regret *exactly* zero, where an ulp of regret with zero gain would
    # flip its ratio from 0 to inf.
    rho_sums = cond_sums[ctx_col, opt_idx, np.arange(m_max)[np.newaxis, :]].sum(axis=1)
    delta = np.clip((rho_sums[:, np.newaxis] - row_sums).T / n_draws, 0.0, None)

    # guard zero-count columns only: split ties leave fractional counts,
    # which clamping up to one would rescale
    nonzero = np.where(counts_sub > 0.0, counts_sub, 1.0)
    mu_cond = cond_sums / nonzero[:, np.newaxis, :]
    # counts == 0 leaves mu_cond at 0, but p_opt == 0 zeroes the term anyway
    dev = mu_cond - means.T[:, :, np.newaxis]
    dev *= dev
    v = np.matmul(dev, p_opt[:, :, np.newaxis])[:, :, 0].T
    return delta, v


def _optimal_two_point(
    delta: NDArray[np.float64],
    v: NDArray[np.float64],
    alive: NDArray[np.bool_],
) -> Tuple[
    NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]
]:
    """Per context, the two-point distribution minimizing the information ratio.

    The minimizer of :math:`\\Delta(\\pi)^2 / v(\\pi)` is supported on at
    most two arms (Russo & Van Roy 2018, Prop. 6), so scanning pairs
    suffices. For a pair with weight ``q`` on ``a`` the ratio is
    ``(A q + B)^2 / (C q + D)`` with ``A = delta_a - delta_b``,
    ``B = delta_b``, ``C = v_a - v_b``, ``D = v_b``; its stationary
    points are ``q = -B / A`` and ``q = (C B - 2 A D) / (A C)``, so the
    minimum over ``[0, 1]`` is at one of those or an endpoint.

    The scan is cut far below ``K^2 x 4`` without excluding a possible
    optimum: the endpoints are the single-arm ratios, whose minimum is
    one ``O(K)`` sweep; ``q = -B / A`` is never strictly interior once
    regret is clipped at zero (``B >= 0`` forces it outside ``(0, 1)``);
    only the Pareto frontier of the ``(v_a, delta_a)`` points can
    support an optimal mixture, since moving weight from a dominated arm
    to its dominator never increases the ratio; and
    ``f(q; a, b) == f(1 - q; b, a)`` halves the pairs. What remains is
    one interior root per unordered frontier pair,
    ``O(K log K + m^2)`` per context with ``m`` typically single digits.

    A mixture with exactly zero regret has ratio 0; positive regret with
    zero gain has ratio ``inf``. Dead arms are excluded.

    Returns
    -------
    a_idx, b_idx, q, ratio : NDArray
        Each ``(n_contexts,)``: play ``a_idx`` with probability ``q``,
        else ``b_idx``, achieving ``ratio``; ``inf`` means the posterior
        has resolved.
    """
    n_contexts = delta.shape[1]
    ctx_idx = np.arange(n_contexts)

    # Endpoints: the best degenerate distribution, O(K * C).
    single = np.full(delta.shape, np.inf)
    np.divide(delta * delta, v, out=single, where=v > 0.0)
    single[delta == 0.0] = 0.0
    single = np.where(alive, single, np.inf)
    s_idx = single.argmin(axis=0)
    s_ratio = single[s_idx, ctx_idx]

    # Pareto frontier: sort by v descending (dead arms last, never
    # dominating), keep arms whose regret undercuts every higher-v arm's
    delta_eff = np.where(alive, delta, np.inf)
    v_key = np.where(alive, v, -1.0)
    order = np.argsort(-v_key, axis=0, kind="stable")  # (K, C)
    d_ord = np.take_along_axis(delta_eff, order, axis=0)
    run_min = np.minimum.accumulate(d_ord, axis=0)
    prev_min = np.empty_like(run_min)
    prev_min[0] = np.inf
    prev_min[1:] = run_min[:-1]
    keep_ord = d_ord < prev_min  # (K, C); dead arms sort behind alive ones
    m_counts = keep_ord.sum(axis=0)
    m_max = int(m_counts.max())
    if m_max < 2:
        return s_idx, s_idx, np.ones(n_contexts), s_ratio

    pos = np.argsort(~keep_ord, axis=0, kind="stable")[:m_max]  # (m, C)
    keep_idx = np.take_along_axis(order, pos, axis=0).T  # (C, m)
    ctx_col = ctx_idx[:, np.newaxis]
    d_front = delta[keep_idx, ctx_col]  # (C, m)
    v_front = v[keep_idx, ctx_col]

    # Ragged pair list: each context contributes only its own frontier's
    # pairs (frontier sizes are heavy-tailed, so one wide context must
    # not inflate the scan for the rest)
    iu_a, iu_b = np.triu_indices(m_max, k=1)
    in_ctx = iu_b[np.newaxis, :] < m_counts[:, np.newaxis]  # (C, n_pairs)
    ctx_rep, pair_rep = np.nonzero(in_ctx)  # row-major: sorted by context
    i_loc = iu_a[pair_rep]
    j_loc = iu_b[pair_rep]
    d_a = d_front[ctx_rep, i_loc]
    d_b = d_front[ctx_rep, j_loc]
    v_a = v_front[ctx_rep, i_loc]
    v_b = v_front[ctx_rep, j_loc]
    a_delta = d_a - d_b
    c_gain = v_a - v_b

    with np.errstate(divide="ignore", invalid="ignore"):
        q = (c_gain * d_b - 2.0 * a_delta * v_b) / (a_delta * c_gain)
    # NaN and out-of-interval roots drop out: a clipped root is an
    # endpoint, already scanned
    interior = (q > 0.0) & (q < 1.0)
    q = np.where(interior, q, 0.0)

    num = (a_delta * q + d_b) ** 2
    den = c_gain * q + v_b
    ratio = np.full_like(num, np.inf)
    np.divide(num, den, out=ratio, where=interior & (den > 0.0))

    # segment minimum per context: ctx_rep is sorted, runs contiguous
    present = np.flatnonzero(m_counts >= 2)
    seg_starts = np.searchsorted(ctx_rep, present)
    seg_min = np.minimum.reduceat(ratio, seg_starts)
    # first achiever per segment (exact equality with the reduction's
    # own output; inf == inf is fine, take_pair discards those below)
    seg_of = np.searchsorted(present, ctx_rep)
    achievers = np.flatnonzero(ratio == seg_min[seg_of])
    _, first = np.unique(seg_of[achievers], return_index=True)
    best = achievers[first]  # one flat pair index per present context

    p_ratio = np.full(n_contexts, np.inf)
    p_ratio[present] = seg_min
    q_ctx = np.zeros(n_contexts)
    q_ctx[present] = q[best]
    a_ctx = np.zeros(n_contexts, dtype=np.intp)
    b_ctx = np.zeros(n_contexts, dtype=np.intp)
    a_ctx[present] = keep_idx[present, i_loc[best]]
    b_ctx[present] = keep_idx[present, j_loc[best]]

    take_pair = p_ratio < s_ratio
    a_idx = np.where(take_pair, a_ctx, s_idx)
    b_idx = np.where(take_pair, b_ctx, s_idx)
    q_best = np.where(take_pair, q_ctx, 1.0)
    return a_idx, b_idx, q_best, np.where(take_pair, p_ratio, s_ratio)


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
    Bayesian bound as the information-ratio analysis of Thompson sampling
    [2]_, while its per-round ratio minimization can be substantially smaller in
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

    #: Information gain conditions on the within-draw argmax (arms must
    #: be jointly distributed) but each context is solved on its own:
    #: exactly ``CONTEXT_JOINT``.
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
        samples = _finite_draws(draw_contiguous(samples), rng)
        n_arms, n_contexts, _ = samples.shape
        n_slots = 1 if top_k is None else min(top_k, n_arms)

        alive = np.ones((n_arms, n_contexts), dtype=np.bool_)
        # masked (dead arms at -inf) and the boolean scratch are
        # maintained across slots instead of re-materialized per slot
        masked = samples if n_slots == 1 else samples.copy()
        indicator = np.empty_like(samples, dtype=np.bool_)
        ctx_idx = np.arange(n_contexts)
        choices = np.empty((n_slots, n_contexts), dtype=np.intp)
        for slot in range(n_slots):
            delta, v = _vids_statistics(
                samples, alive, masked=masked, indicator=indicator
            )
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

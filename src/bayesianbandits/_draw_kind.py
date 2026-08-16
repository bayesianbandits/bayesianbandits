"""What a policy needs of its posterior draws.

A policy consumes draws; an agent supplies them. The two are matched by
a single value describing the *weakest* draws a policy can correctly
consume, rather than by flags naming the sampling methods it will
tolerate. That inversion is the point: a policy states what it needs,
an agent picks the cheapest exact way to provide it, and neither has to
know the other's implementation.
"""

from enum import IntEnum

__all__ = ["DrawKind"]


class DrawKind(IntEnum):
    """How much joint structure a policy's decisions depend on.

    The three levels are totally ordered by how much dependence they
    preserve, so they compare and combine with ``<`` and :func:`max`:

    .. code-block:: text

        MARGINAL_ONLY  <  CONTEXT_JOINT  <  JOINT

    Supplying *more* structure than a policy asks for is always sound:
    the requested statistics keep the same distribution, only the
    dependence a policy does not read gets preserved anyway. Supplying
    less never is. So an agent may always widen, and must never narrow.
    Widening silently is fine; narrowing must be impossible, which is
    what stating the requirement (rather than permitting a method)
    buys.

    Cheapness runs the other way, ``MARGINAL_ONLY`` being cheapest, so a
    policy that overstates its needs only pays for it, and one that
    understates them is wrong. When in doubt, use ``JOINT``.
    """

    MARGINAL_ONLY = 0
    """Only per-(arm, context) statistics are read.

    Each cell may be drawn independently, so ``sample_marginal`` serves.
    Correct for policies scoring every arm on its own -- a quantile, a
    mean, an upper confidence bound -- and wrong for any that compares
    arms *within* a draw, since independent cells carry no such
    comparison.
    """

    CONTEXT_JOINT = 1
    """Arms are compared within a context, but contexts never interact.

    Draws must be jointly distributed across the arms of one context
    and may be independent across contexts. This is what
    ``sample_reward_space(block_size=n_arms)`` produces, and what
    information-directed sampling needs: it reads the covariance
    between arms under a shared posterior, but solves each context's
    decision on its own.

    Note that a ``NormalInverseGammaRegressor`` under ``JOINT`` shares
    one noise-scale draw across contexts, and under ``CONTEXT_JOINT``
    does not. Both are exact for their own level; only a policy that
    couples contexts can tell them apart.
    """

    JOINT = 2
    """Every arm and context shares one posterior draw.

    The strongest requirement and the safe default: one weight vector
    per draw, so every dependence the posterior implies is present.
    Thompson sampling needs it, and so does any reward transform that
    combines arms before the policy sees them.
    """

"""Agent-wrapping pipeline implementation for Bayesian bandits.

This module implements agent-wrapping pipelines that apply preprocessing steps
before delegating to a wrapped ContextualAgent. This enables efficient
preprocessing at the agent level rather than per-arm.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .._arm import Arm, ContextType, TokenType
from .._memory import MemoryUsageMixin
from ..api import Agent, ContextualAgent, PolicyProtocol


def _validate_steps(steps: List[Tuple[str, Any]]) -> None:
    """Validate pipeline steps."""
    if not steps:
        raise ValueError("Pipeline steps cannot be empty")

    names, _ = zip(*steps)

    # Validate names are unique
    if len(set(names)) != len(names):
        raise ValueError("Step names must be unique")


def _transform_data(X: Any, steps: List[Tuple[str, Any]]) -> Any:
    """Apply all transformers to input data.

    Transformers must be either stateless or pre-fitted.
    No fitting occurs during transformation.
    """
    result = X

    for name, transformer in steps:
        try:
            result = transformer.transform(result)
        except Exception as e:
            # Provide helpful error for common case
            if hasattr(e, "args") and "not fitted" in str(e).lower():
                raise RuntimeError(
                    f"Transformer '{name}' is not fitted. In online learning, "
                    f"all transformers must be either stateless or pre-fitted "
                    f"before use. Common stateless transformers include "
                    f"FunctionTransformer, FeatureHasher, and HashingVectorizer. "
                    f"Stateful transformers like StandardScaler must be fit on "
                    f"historical data before creating the pipeline."
                ) from e
            raise

    return result


class AgentPipeline(MemoryUsageMixin, Generic[ContextType, TokenType]):
    """Pipeline that wraps a ContextualAgent.

    Transforms input data through preprocessing steps before delegating
    to the wrapped ContextualAgent. The input can be any type that the
    first transformer accepts, and the output of the transformation chain
    must match the ContextType expected by the agent.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transformer) tuples for preprocessing steps.
        The final output must match ContextType for the agent.
    final_agent : ContextualAgent[ContextType, TokenType]
        The ContextualAgent to wrap and delegate to.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> from bayesianbandits import Arm, NormalRegressor, ContextualAgent, ThompsonSampling
    >>>
    >>> # Create arms and agent expecting sparse arrays
    >>> arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True)) for i in range(3)]
    >>> agent = ContextualAgent(arms, ThompsonSampling())
    >>>
    >>> # Pipeline can accept dict input and transform to sparse
    >>> vectorizer = DictVectorizer(sparse=True)
    >>> _ = vectorizer.fit([{'user': 'A', 'item': 1}, {'user': 'B', 'item': 2}])
    >>> pipeline = AgentPipeline(
    ...     steps=[('vectorize', vectorizer)],
    ...     final_agent=agent
    ... )
    >>>
    >>> # Input is dict, output is sparse matrix for agent
    >>> X_dict = [{'user': 'A', 'item': 1}]
    >>> recommendations = pipeline.pull(X_dict)
    >>> pipeline.update(X_dict, np.array([1.0]))
    """

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        final_agent: ContextualAgent[ContextType, TokenType],
    ) -> None:
        if isinstance(final_agent, Agent):
            raise TypeError(
                "AgentPipeline wraps a contextual agent, whose context the "
                "steps transform. A non-contextual Agent has no context to "
                "transform, so the steps would never run. Preprocess the "
                "arms' features with a LearnerPipeline instead."
            )
        _validate_steps(steps)
        self.steps = steps
        self._agent = final_agent

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline steps by name."""
        return dict(self.steps)

    def transform(self, X: Any) -> Any:
        """Apply all transformers to input data."""
        return _transform_data(X, self.steps)

    @overload
    def pull(self, X: Any) -> List[TokenType]: ...

    @overload
    def pull(self, X: Any, *, top_k: int) -> List[List[TokenType]]: ...

    def pull(
        self, X: Any, *, top_k: Optional[int] = None
    ) -> Union[List[TokenType], List[List[TokenType]]]:
        """Choose arm(s) and pull based on the context(s).

        Parameters
        ----------
        X : Any
            Input data to transform and use for choosing arms.
            Will be transformed through the pipeline steps to ContextType.
        top_k : int, optional
            Number of arms to select per context. If None (default),
            selects single best arm per context.

        Returns
        -------
        List[TokenType] or List[List[TokenType]]
            If top_k is None: List of action tokens (one per context)
            If top_k is int: List of lists of action tokens
        """
        X_transformed = self.transform(X)
        if top_k is None:
            return self._agent.pull(X_transformed)
        else:
            return self._agent.pull(X_transformed, top_k=top_k)

    def update(
        self,
        X: Any,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the wrapped agent with context(s) and reward(s).

        Parameters
        ----------
        X : Any
            Input data to transform and use for updating the arm.
            Will be transformed through the pipeline steps to ContextType.
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        sample_weight : Optional[NDArray[np.float64]], default=None
            Sample weights to use for updating the arm.
        """
        X_transformed = self.transform(X)
        self._agent.update(X_transformed, y, sample_weight=sample_weight)

    def decay(
        self,
        forgetting: Any = None,
        *,
        decay_rate: Optional[float] = None,
        steps: float = 1,
    ) -> None:
        """Forget on every arm of the wrapped agent: the clock ticked.

        Parameters
        ----------
        forgetting : ExponentialForgetting or StabilizedForgetting, optional
            The rule to tick with, carrying its own rate.
        steps : float, default=1
            Number of ticks; the rule's rate is raised to this power.
        decay_rate : float, optional
            Shorthand for each learner's default rule at this rate.
        """
        self._agent.decay(forgetting, decay_rate=decay_rate, steps=steps)

    # Delegation methods
    def add_arm(self, arm: Arm[Any, TokenType]) -> None:
        """Add an arm to the wrapped agent."""
        self._agent.add_arm(arm)

    def remove_arm(self, token: TokenType) -> None:
        """Remove an arm from the wrapped agent."""
        self._agent.remove_arm(token)

    def arm(self, token: TokenType) -> Arm[Any, TokenType]:
        """Get an arm by its action token."""
        return self._agent.arm(token)

    def select_for_update(self, token: TokenType) -> Self:
        """Set the arm to update and return self for chaining."""
        self._agent.select_for_update(token)
        return self

    @property
    def arms(self) -> List[Arm[ContextType, TokenType]]:
        """Get the arms from the wrapped agent."""
        return self._agent.arms

    @property
    def arm_to_update(self) -> Arm[ContextType, TokenType]:
        """Get the arm to update from the wrapped agent."""
        return self._agent.arm_to_update

    @property
    def policy(self) -> PolicyProtocol[ContextType, TokenType]:
        """Get the policy from the wrapped agent."""
        return self._agent.policy

    @policy.setter
    def policy(self, value: PolicyProtocol[ContextType, TokenType]) -> None:
        """Set the policy on the wrapped agent."""
        self._agent.policy = value

    @property
    def rng(self) -> np.random.Generator:
        """Get the random generator from the wrapped agent."""
        return self._agent.rng

    @rng.setter
    def rng(self, value: Union[int, None, np.random.Generator]) -> None:
        """Set the random generator on the wrapped agent."""
        self._agent.rng = value

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.steps
        ]
        return f"AgentPipeline(steps=[{', '.join(steps_repr)}], final_agent={self._agent!r})"

    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str]) -> Any:
        """Get a step by index or name."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind]


#: Kept for backward compatibility. One pipeline class serves every
#: agent that has a context for its steps to transform.
ContextualAgentPipeline = AgentPipeline

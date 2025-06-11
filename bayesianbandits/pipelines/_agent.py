"""Agent-wrapping pipeline implementation for Bayesian bandits.

This module implements agent-wrapping pipelines that apply preprocessing steps
before delegating to wrapped Agent/ContextualAgent instances. This enables
efficient preprocessing at the agent level rather than per-arm.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .._arm import ContextType, TokenType
from ..api import Agent, ContextualAgent


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


class ContextualAgentPipeline(Generic[ContextType, TokenType]):
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
    >>> pipeline = ContextualAgentPipeline(
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
        X: Any,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay all arms of the wrapped agent.

        Parameters
        ----------
        X : Any
            Input data to transform and use for decaying the arms.
            Will be transformed through the pipeline steps to ContextType.
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arms.
        """
        X_transformed = self.transform(X)
        self._agent.decay(X_transformed, decay_rate=decay_rate)

    # Delegation methods
    def add_arm(self, arm) -> None:
        """Add an arm to the wrapped agent."""
        self._agent.add_arm(arm)

    def remove_arm(self, token: TokenType) -> None:
        """Remove an arm from the wrapped agent."""
        self._agent.remove_arm(token)

    def arm(self, token: TokenType):
        """Get an arm by its action token."""
        return self._agent.arm(token)

    def select_for_update(self, token: TokenType) -> Self:
        """Set the arm to update and return self for chaining."""
        self._agent.select_for_update(token)
        return self

    @property
    def arms(self):
        """Get the arms from the wrapped agent."""
        return self._agent.arms

    @property
    def arm_to_update(self):
        """Get the arm to update from the wrapped agent."""
        return self._agent.arm_to_update

    @property
    def policy(self):
        """Get the policy from the wrapped agent."""
        return self._agent.policy

    @policy.setter
    def policy(self, value):
        """Set the policy on the wrapped agent."""
        self._agent.policy = value

    @property
    def rng(self):
        """Get the random generator from the wrapped agent."""
        return self._agent.rng

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.steps
        ]
        return f"ContextualAgentPipeline(steps=[{', '.join(steps_repr)}], final_agent={self._agent!r})"

    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str]) -> Any:
        """Get a step by index or name."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind]


class NonContextualAgentPipeline(Generic[TokenType]):
    """Pipeline that wraps an Agent.

    For non-contextual agents, preprocessing steps are not applied since
    there's no context to transform. This class exists primarily for
    API consistency and to support future extensions.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transformer) tuples. For non-contextual agents,
        these are typically unused but kept for API consistency.
    final_agent : Agent[TokenType]
        The Agent to wrap and delegate to.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits import Arm, NormalRegressor, Agent, ThompsonSampling
    >>>
    >>> # Create arms and agent
    >>> arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)]
    >>> agent = Agent(arms, ThompsonSampling())
    >>>
    >>> # Create pipeline (steps are unused for non-contextual)
    >>> pipeline = NonContextualAgentPipeline(
    ...     steps=[],  # No preprocessing needed
    ...     final_agent=agent
    ... )
    >>>
    >>> # Use like a normal Agent
    >>> recommendations = pipeline.pull()
    >>> pipeline.update(np.array([1.0]))
    """

    def __init__(
        self, steps: List[Tuple[str, Any]], final_agent: Agent[TokenType]
    ) -> None:
        _validate_steps(
            steps
        ) if steps else None  # Allow empty steps for non-contextual
        self.steps = steps
        self._agent = final_agent

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline steps by name."""
        return dict(self.steps)

    @overload
    def pull(self) -> List[TokenType]: ...

    @overload
    def pull(self, *, top_k: int) -> List[List[TokenType]]: ...

    def pull(
        self, *, top_k: Optional[int] = None
    ) -> Union[List[TokenType], List[List[TokenType]]]:
        """Choose arm(s) and pull.

        Parameters
        ----------
        top_k : int, optional
            Number of arms to select. If None (default), selects single
            best arm.

        Returns
        -------
        List[TokenType] or List[List[TokenType]]
            If top_k is None: List containing single action token
            If top_k is int: List containing list of action tokens
        """
        if top_k is None:
            return self._agent.pull()
        else:
            return self._agent.pull(top_k=top_k)

    def update(
        self,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the wrapped agent with observed reward(s).

        Parameters
        ----------
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        sample_weight : Optional[NDArray[np.float64]], default=None
            Sample weights to use for updating the arm.
        """
        self._agent.update(y, sample_weight=sample_weight)

    def decay(self, decay_rate: Optional[float] = None) -> None:
        """Decay all arms of the wrapped agent.

        Parameters
        ----------
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arms.
        """
        self._agent.decay(decay_rate=decay_rate)

    # Delegation methods
    def add_arm(self, arm) -> None:
        """Add an arm to the wrapped agent."""
        self._agent.add_arm(arm)

    def remove_arm(self, token: TokenType) -> None:
        """Remove an arm from the wrapped agent."""
        self._agent.remove_arm(token)

    def arm(self, token: TokenType):
        """Get an arm by its action token."""
        return self._agent.arm(token)

    def select_for_update(self, token: TokenType) -> Self:
        """Set the arm to update and return self for chaining."""
        self._agent.select_for_update(token)
        return self

    @property
    def arms(self):
        """Get the arms from the wrapped agent."""
        return self._agent.arms

    @property
    def arm_to_update(self):
        """Get the arm to update from the wrapped agent."""
        return self._agent.arm_to_update

    @property
    def policy(self):
        """Get the policy from the wrapped agent."""
        return self._agent.policy

    @policy.setter
    def policy(self, value):
        """Set the policy on the wrapped agent."""
        self._agent.policy = value

    @property
    def rng(self):
        """Get the random generator from the wrapped agent."""
        return self._agent.rng

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.steps
        ]
        return f"NonContextualAgentPipeline(steps=[{', '.join(steps_repr)}], final_agent={self._agent!r})"

    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str]) -> Any:
        """Get a step by index or name."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind]


# Factory function with overloads
@overload
def AgentPipeline(
    steps: List[Tuple[str, Any]], final_agent: ContextualAgent[ContextType, TokenType]
) -> ContextualAgentPipeline[ContextType, TokenType]: ...


@overload
def AgentPipeline(
    steps: List[Tuple[str, Any]], final_agent: Agent[TokenType]
) -> NonContextualAgentPipeline[TokenType]: ...


def AgentPipeline(
    steps: List[Tuple[str, Any]],
    final_agent: Union[ContextualAgent[ContextType, TokenType], Agent[TokenType]],
) -> Union[
    ContextualAgentPipeline[ContextType, TokenType],
    NonContextualAgentPipeline[TokenType],
]:
    """Create a Pipeline that wraps an Agent or ContextualAgent.

    This factory function provides a clean API for creating pipelines
    while maintaining complete static typing based on the agent type.
    The pipeline can accept any input type and transform it to what the agent expects.

    The resulting Pipeline will have the same interface as the wrapped agent,
    allowing you to call `pull`, `update`, and other methods directly on it.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transformer) tuples for preprocessing steps.
        All transformers must be either stateless or pre-fitted.
        The output of the transformation chain must match the agent's expected input type.
    final_agent : Agent[TokenType] or ContextualAgent[ContextType, TokenType]
        The agent to wrap. The pipeline type is determined by the agent type.

    Returns
    -------
    ContextualAgentPipeline or NonContextualAgentPipeline
        The appropriate pipeline type based on the final_agent type.

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> from bayesianbandits import Arm, NormalRegressor, ContextualAgent, ThompsonSampling
    >>>
    >>> # Pipeline accepting dict input, outputting sparse arrays for agent
    >>> arms = [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True)) for i in range(3)]
    >>> agent = ContextualAgent(arms, ThompsonSampling())
    >>> vectorizer = DictVectorizer()
    >>> _ = vectorizer.fit([{'user': 'A'}, {'user': 'B'}])
    >>>
    >>> pipeline = AgentPipeline(
    ...     steps=[('vectorize', vectorizer)],
    ...     final_agent=agent
    ... )
    >>> # Can accept dict input: [{'user': 'A', 'item': 1}]
    >>> # Transforms to sparse matrix for agent
    """
    if isinstance(final_agent, Agent):
        return NonContextualAgentPipeline(steps, final_agent)
    return ContextualAgentPipeline(steps, final_agent)

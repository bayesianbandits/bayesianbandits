"""Pipeline utilities for bayesian bandits.

This module provides two types of pipelines:

1. **AgentPipeline**: Wraps Agent/ContextualAgent with preprocessing steps.
   Use this when you want to apply sklearn preprocessing to raw input data
   before it reaches the agent.

2. **LearnerPipeline**: Implements the Learner protocol with sklearn preprocessing.
   Use this as a learner within Arms when you want to apply sklearn preprocessing
   to enriched features (e.g., after ArmFeaturizer transforms).
"""

from ._agent import (
    AgentPipeline,
    ContextualAgentPipeline,
    NonContextualAgentPipeline,
)
from ._learner import LearnerPipeline

__all__ = [
    "AgentPipeline",
    "ContextualAgentPipeline",
    "NonContextualAgentPipeline",
    "LearnerPipeline",
]

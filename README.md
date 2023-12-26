# `bayesianbandits` [![Downloads](https://static.pepy.tech/badge/bayesianbandits/month)](https://pepy.tech/project/bayesianbandits)

## Bayesian Multi-Armed Bandits for Python

**Problem**: Despite having a conceptually simple interface, putting together a multi-armed bandit in Python is a daunting task. 

**Solution**: `bayesianbandits` is a Python package that provides a simple interface for creating and running Bayesian multi-armed bandits. It is built on top of [scikit-learn](https://scikit-learn.org/stable/) and [scipy](https://www.scipy.org/), taking advantage of conjugate priors to provide fast and accurate inference.

While the API is still evolving, this library is already being used in production for marketing optimization, dynamic pricing, and other applications. Are you using `bayesianbandits` in your project? Let us know!

## Features

* **Simple API**: `bayesianbandits` provides a simple interface - most users will only need to call `pull` and `update` to get started.
* **Fast**: `bayesianbandits` is built on top of already fast scientific Python libraries, but, if installed, will also use SuiteSparse to further speed up matrix operations on sparse matrices. Handling tens or even hundreds of thousands of features in a sparse model is no problem.
* **scikit-learn compatible**: Use sklearn pipelines and transformers to preprocess data before feeding it into your bandit.
* **Flexible**: Pick from a variety of policy algorithms, including Thompson sampling, upper confidence bound, and epsilon-greedy. Pick from a variety of prior distributions, including beta, gamma, normal, and normal-inverse-gamma.
* **Extensible**: `bayesianbandits` provides simple interfaces for creating custom policies and priors.
* **Well-tested**: `bayesianbandits` is well-tested, with nearly 100% test coverage.

## Getting Started

Install this package from PyPI.

```
pip install -U bayesianbandits
```

Define a LinearUCB contextual bandit with a normal prior.

```python
import numpy as np
from bayesianbandits import (
    Arm,
    NormalInverseGammaRegressor,
)
from bayesianbandits.api import (
    ContextualAgent,
    UpperConfidenceBound,
)

arms = [
    Arm(1, learner=NormalInverseGammaRegressor()),
    Arm(2, learner=NormalInverseGammaRegressor()),
    Arm(3, learner=NormalInverseGammaRegressor()),
    Arm(4, learner=NormalInverseGammaRegressor()),
]

policy = UpperConfidenceBound(alpha=0.84)
```

Instantiate the agent and pull an arm with context.

```python
agent = ContextualAgent(arms, policy)

context = np.array([[1, 0, 0, 0]])

# Can be constructed with sklearn, formulaic, patsy, etc...
# context = formulaic.Formula("1 + article_number").get_model_matrix(data)
# context = sklearn.preprocessing.OneHotEncoder().fit_transform(data)

agent.pull(context)
```

Update the bandit with the reward.

```python
agent.update(context, np.array([15.0]))
```

That's it! Check out the [documentation](https://bayesianbandits.readthedocs.io/en/latest/) for more examples.

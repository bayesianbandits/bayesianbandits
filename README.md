# `bayesianbandits` [![Downloads](https://static.pepy.tech/badge/bayesianbandits/month)](https://pepy.tech/project/bayesianbandits)

## Bayesian Multi-Armed Bandits for Python

**Problem**: Despite having a conceptually simple interface, putting together a multi-armed bandit in Python is a daunting task. 

**Solution**: `bayesianbandits` is a Python package that provides a simple interface for creating and running Bayesian multi-armed bandits. It is built on top of [scikit-learn](https://scikit-learn.org/stable/) and [scipy](https://www.scipy.org/), taking advantage of conjugate priors to provide fast and accurate inference.

While the API is still evolving, this library is already being used in production for marketing optimization, dynamic pricing, and other applications. Are you using `bayesianbandits` in your project? Let us know!

## Features

* **Simple API**: `bayesianbandits` provides a simple interface - most users will only need to call `pull` and `update` to get started.
* **Fast**: `bayesianbandits` is built on top of [scikit-learn](https://scikit-learn.org/stable/) and [scipy](https://www.scipy.org/), taking advantage of conjugate priors to provide fast and accurate inference. If present, `bayesianbandits` will use SuiteSparse to speed up matrix operations on sparse matrices.
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
    Bandit,
    NormalInverseGammaRegressor,
    upper_confidence_bound,
    contextual,
)

est = NormalInverseGammaRegressor(lam=4)
policy = upper_confidence_bound(alpha=0.80, samples=500)

@contextual
class Agent(Bandit, learner=est, policy=policy):
    article_1 = Arm(1)
    article_2 = Arm(2)
    article_3 = Arm(3)
    article_4 = Arm(4)

```

Instantiate the bandit and pull an arm with context.

```python
agent = Agent()

context = np.array([[1, 0, 0, 0]])

# Can be constructed with sklearn, formulaic, patsy, etc...
# context = formulaic.Formula("1 + article_number").get_model_matrix(data)
# context = sklearn.preprocessing.OneHotEncoder().fit_transform(data)

agent.pull(context)
```

Update the bandit with the reward.

```python
agent.update(context, 15.0)
```

That's it! Check out the [documentation](https://bayesianbandits.readthedocs.io/en/latest/) for more examples.

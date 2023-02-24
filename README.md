# `bayesianbandits`

bayesianbandits is a Pythonic framework for building agents to maximize rewards in multi-armed bandit (MAB) problems. These agents can handle a number of MAB subproblems, such as contextual, restless, and delayed reward bandits.

Building an agent is as simple as defining arms and using the necessary decorators. For example, to create an agent for a Bernoulli bandit:

```python
import numpy as np

from bayesianbandits import contextfree, bandit, epsilon_greedy, Arm, DirichletClassifier

def reward_func(x):
    return np.take(x, 0, axis=-1)

def action1_func():
    # do action 1
    ...

def action2_func():
    # do action 2
    ...

@bandit(learner=DirichletClassifier({"yes": 1.0, "no": 1.0}), policy=epsilon_greedy(0.1))
class Agent:
    arm1 = Arm(action1_func, reward_func)
    arm2 = Arm(action2_func, reward_func)

agent = Agent()

agent.pull() # receive some reward
agent.update("yes") # update with observed reward

```

## Getting Started

Install this package from PyPI.

```
pip install -U bayesianbandits
```

## Usage

Check out the [documentation](https://bayesianbandits.readthedocs.io/en/latest/) for examples and an API reference. 
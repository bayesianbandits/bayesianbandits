# Documentation Strategy

## Audiences

1. **Practitioners** -- want to solve a bandit problem, need to know which knobs to turn
2. **Researchers/skeptics** -- want to verify the math before trusting it in production
3. **LLMs** -- acting as intermediaries (Copilot, Claude, etc.), need self-contained structured text to synthesize correct answers

## Site Outline

```
index.rst                                    (restructured toctree)
|
+-- Getting Started
|   +-- installation.rst                     exists
|   +-- introduction.rst                     exists (Explanation)
|   +-- quickstart.rst                       exists (Tutorial)
|
+-- How-To Guides
|   +-- howto/pipelines.rst                  DONE - Integrating with sklearn transformers
|   +-- howto/decay.rst                      DONE - Choosing and tuning a decay rate
|   +-- howto/reward-functions.rst           Writing custom reward functions
|   |                                        (absorbs demo.ipynb content)
|   +-- howto/delayed-rewards.rst            Handling delayed rewards
|   +-- howto/production.rst                 Deploying a bandit to production
|   |                                        (absorbs persistence.ipynb content)
|   +-- howto/sparse.rst                     Working with sparse features / CHOLMOD
|
+-- Mathematical Reference                   ALL NEW
|   +-- math/normal.rst                      NormalRegressor: prior, update, decay
|   +-- math/normal-inverse-gamma.rst        NIG: joint posterior, marginal t
|   +-- math/empirical-bayes.rst             MacKay EB, stabilized forgetting
|   +-- math/intercept-only.rst              Dirichlet-Multinomial, Gamma-Poisson
|   |                                        (absorbs counts.ipynb content)
|   +-- math/glm.rst                         Laplace approximation, IRLS
|   +-- math/policies.rst                    TS, UCB, epsilon-greedy, EXP3
|
+-- Examples (Cookbook)                       6 notebooks, down from 9
|   +-- notebooks/linear-bandits             NIG vs GLM with regret curves
|   +-- notebooks/hybrid-bandits             Design matrix structures
|   +-- notebooks/empirical-bayes            MacKay convergence + stabilized forgetting
|   +-- notebooks/adversarial                EXP3A, Nash equilibrium
|   +-- notebooks/delayed-reward             Non-stationary + optuna tuning
|   +-- notebooks/offline-learning           Warm-start from historical data
|
+-- API Reference
|   +-- api.rst                              exists (needs docstring improvements)
|
+-- changelog.rst                            nice-to-have
```

## Content Absorption Plan

Three notebooks are cut; their unique content moves to how-to or math pages:

| Cut Notebook | Content Destination |
|-------------|---------------------|
| `demo.ipynb` | Custom reward functions go to `howto/reward-functions.rst`. Batch updates go to `howto/production.rst`. |
| `counts.ipynb` | GammaRegressor math goes to `math/intercept-only.rst`. UCB tuning guidance goes to `math/policies.rst`. |
| `persistence.ipynb` | Joblib serialization, adding/removing arms go to `howto/production.rst`. |

## Structure: Three Layers

### Layer 1: User Guide (narrative, opinionated)

Follows the [Diataxis](https://diataxis.fr/) framework. Pages are typed by purpose:

**Explanation** (understanding-oriented, no code)

- **Introduction** (`introduction.rst`): When to use a bandit, why Bayesian, why conjugate models, and a decision guide mapping problem shapes to (estimator, agent, policy) combinations. Prose only. Done.

**Tutorials** (learning-oriented, complete code)

- **Quick Start** (`quickstart.rst`): Get a working bandit in 5 minutes. A single end-to-end example with the pull/update loop. Done.

**How-to guides** (task-oriented, code snippets)

Short pages (1-2 pages each) for specific tasks:

- `howto/pipelines.rst` -- Using sklearn pipelines with bandits
- `howto/decay.rst` -- Choosing and tuning a decay rate
- `howto/reward-functions.rst` -- Writing custom reward functions
- `howto/delayed-rewards.rst` -- Handling delayed rewards
- `howto/production.rst` -- Deploying a bandit to production
- `howto/sparse.rst` -- Working with sparse features / CHOLMOD

**Reference**

Covered by Layer 2 (math) and Layer 3 (API docs) below.

Notebooks are demoted to an "Examples" section, supplementary, not primary.

### Layer 2: Mathematical Reference (rigorous, static)

Concise style: state equations, explain parameterization, link to Murphy/Bishop for derivations. Use numbered LaTeX equations via MathJax.

Each page follows a consistent structure:

1. Symbol definitions table
2. Generative model (likelihood + prior)
3. Update equations (posterior)
4. Hyperparameter semantics (what each controls, practical ranges)
5. Departures from textbook (if any)
6. How decay interacts with the model

Six pages: one per estimator family + one for policies.

### Layer 3: API Reference (structured for machines)

Auto-generated from docstrings via autodoc/autosummary. Quality depends entirely on docstring quality.

Key principles for LLM-parseable docstrings:
- **Every public class/method has a complete docstring** -- not "see the user guide." LLMs often see docstrings in isolation.
- **Parameters section is precise about types, shapes, and semantics.** Not just `alpha : float` but `alpha : float -- Prior precision. Higher values mean stronger regularization toward zero.`
- **Returns section with shapes.** Especially for `sample()` and `predict()`.
- **Examples in every class docstring.** A 5-line usage example in the docstring is worth more for LLM parsing than a 200-cell notebook.
- **Consistent structure.** Every class follows: one-line summary, extended summary, Parameters, Attributes, Returns (for methods), Examples, See Also, Notes.
- **See Also sections** linking related classes with a one-line note about when to prefer each.

## Notebooks

Notebooks are **examples**, not documentation. They belong in an "Examples" section and can be exploratory/narrative without being comprehensive.

Figures from notebooks that illustrate a concept (posterior distributions updating, regret curves) can be pulled out as static images in `docs/_static/` and referenced from the prose pages. This decouples the docs from the notebooks.

For the math reference, purpose-built figures (clean matplotlib with consistent styling) will communicate better than notebook screenshots with scenario-specific axis labels.

## Docstring Audit

### Current state (as of 2026-03-17)

**Estimator classes** (DirichletClassifier, GammaRegressor, NormalRegressor, NormalInverseGammaRegressor, BayesianGLM, EmpiricalBayesNormalRegressor): All have complete docstrings with Parameters, Attributes, See Also, Notes (with equations), References, and Examples. No changes needed.

**Policy classes**: Have Parameters, Notes (with regret bounds), and References. Missing See Also and Examples (except EXP3A which has both).

**Pipeline/featurizer classes**: ArmColumnFeaturizer, ContextualAgentPipeline, NonContextualAgentPipeline are missing Notes and See Also. LearnerPipeline is complete.

### Remaining issues to fix

#### 1. Policy See Also and Examples
Add See Also cross-references and usage examples to `ThompsonSampling`, `UpperConfidenceBound`, and `EpsilonGreedy`.

Cross-reference groups:
- `ThompsonSampling` <-> `UpperConfidenceBound` <-> `EpsilonGreedy` <-> `EXP3A`

#### 2. `alpha`/`beta` overloading
`alpha` means prior precision (`NormalRegressor`), confidence level (`UpperConfidenceBound`), and shape parameter (`GammaRegressor`). Docstrings define them locally (correct) but don't acknowledge the overloading.

**Fix:** Add brief notes clarifying parameterization conventions where ambiguous.

## How-To Guide Specifications

Each how-to should be 1-2 pages, task-oriented, with code snippets (not full scenarios).

### `howto/decay.rst` -- Choosing and Tuning a Decay Rate (DONE)
- Default advice: start with `learning_rate=1.0`, no decay
- Decouple decay from updates: recsys example, `decay()` on a schedule
- Note box: what decay does mechanically (precision scaling, re-exploration)
- Footnote: why n_rows exponent exists, why per-obs decay is often too aggressive
- Effective window size table (gamma → window)
- Avoid over-decay: near-singular precision, prior collapse, EB as safety net
- Cross-reference to delayed-reward notebook (optuna tuning)
- Tests: `tests/test_howto_decay.py` (5 tests)

### `howto/pipelines.rst` -- Integrating with sklearn Transformers (DONE)
- `AgentPipeline` for `Agent`/`ContextualAgent` (per-arm learners): accepts JSON (DictVectorizer), DataFrames (ColumnTransformer), or raw arrays (StandardScaler)
- `LearnerPipeline` for `LipschitzContextualAgent` (shared learner): wraps the shared learner, transforms run after the arm featurizer enriches the context
- Critical constraint: all transformers must be stateless or pre-fitted (explained with scaler rationale)
- Two LearnerPipeline design matrix examples: disjoint (one-hot arms) and hybrid (shared context + per-arm offsets)
- Cross-reference to hybrid-bandits notebook for full treatment
- Tests: `tests/test_howto_pipelines.py` (7 tests)

### `howto/sparse.rst` -- Working with Sparse Features
- When to enable `sparse=True`
- CHOLMOD vs. SuperLU backends
- Memory and performance characteristics
- Code: sparse context matrix with 1M features

### `howto/production.rst` -- Deploying to Production
- Serialization with joblib (from persistence.ipynb)
- **Reseeding the RNG after loading**: after deserialization, the RNG state is frozen from save time. Must reseed to avoid deterministic replay of the same exploration sequence. Show how to set `random_state` on the agent/arms after `joblib.load()`.
- Adding/removing arms at runtime
- Batch pull/update for throughput
- Latency characteristics (O(d^2) updates, Cholesky sampling)

### `howto/reward-functions.rst` -- Custom Reward Functions
- Traditional `f(samples) -> rewards`
- Context-aware `f(samples, context) -> rewards`
- Batch signatures for LipschitzContextualAgent
- Code: profit = revenue - cost pattern (from demo.ipynb)

### `howto/delayed-rewards.rst` -- Handling Delayed Rewards
- The core pattern: pull() now, update() later (hours/days)
- Tracking which arm was pulled (storing the action token with the request)
- Batch update when rewards arrive (multiple observations at once)
- Interaction with decay: if decay is on, observations that arrive late are still "current" data but the model has already decayed
- When delayed rewards matter vs. when they don't (stationary vs. non-stationary)
- Code: pull with request ID, collect rewards asynchronously, batch update

## Math Reference Specifications

Each page follows a consistent structure:
1. Symbol definitions table
2. Generative model (likelihood + prior)
3. Update equations (posterior)
4. Hyperparameter semantics (what each controls, practical ranges)
5. Departures from textbook (if any)
6. How decay interacts with the model

Concise style: state equations, explain parameterization, link to Murphy/Bishop for derivations. Use numbered LaTeX equations via MathJax.

### `math/normal.rst`
- Gaussian prior on weights, known noise precision beta
- Precision matrix update: Lambda_new = gamma^n * Lambda_old + beta * X^T W X
- Mean update via precision-weighted observations
- Decay mechanics: gamma < 1 scales precision, increasing posterior variance
- Parameterization: alpha = prior precision, beta = noise precision

### `math/normal-inverse-gamma.rst`
- Joint NIG prior over (weights, noise variance)
- Update equations for all four NIG parameters
- Marginal t-distribution for weights
- Heavier tails with little data

### `math/empirical-bayes.rst`
- MacKay's evidence maximization for alpha and beta
- Online MacKay step in partial_fit
- Stabilized forgetting (Kulhavy-Zarrop): what it does and why
- Precision correction after EB updates

### `math/intercept-only.rst`
- Dirichlet-Multinomial: conjugate update for class probabilities
- Gamma-Poisson: conjugate update for rate parameter
- When to use each

### `math/glm.rst`
- Laplace approximation via IRLS
- Logit and log link functions
- Posterior is Gaussian at the MAP, not exact conjugate
- Trade-offs vs. conjugate models

### `math/policies.rst`
- Thompson sampling: sample from posterior, argmax
- UCB: alpha-quantile of posterior, argmax
- Epsilon-greedy: exploit with 1-eps, explore uniformly
- EXP3A: exponential weights, importance weighting, forced exploration

## Testing Code Examples

Every how-to guide with code snippets must have a corresponding test file in `tests/`, following the pattern established by `tests/test_quickstart.py`:

- Mirror each code snippet as a test function
- Assert the snippets produce expected results (not just "don't crash")
- Test file name matches the how-to: `howto/pipelines.rst` -> `tests/test_howto_pipelines.py`

## Priority Order

### Tier 0: Foundation (DONE)
1. ~~Update `docs/DOCUMENTATION_STRATEGY.md`~~
2. ~~Restructure `docs/index.rst` toctree~~ (hub-and-spoke: section index pages for clean navbar)
3. ~~Add `sphinx.ext.mathjax` to `docs/conf.py`~~
4. ~~Create `docs/howto/` and `docs/math/` directories~~
5. ~~Exclude cut notebooks in `conf.py`, remove `usage.rst`~~
6. ~~Fix `quickstart.rst` cross-references to cut notebooks~~
7. ~~Shorten navbar title via `html_short_title` and `logo.text`~~

### Tier 1: Highest leverage
8. ~~`howto/pipelines.rst` + `tests/test_howto_pipelines.py`~~ (DONE)
9. ~~`howto/decay.rst` + `tests/test_howto_decay.py`~~ (DONE)
10. Docstring audit (policy See Also + Examples)
11. `howto/reward-functions.rst` + `tests/test_howto_reward_functions.py`

### Tier 2: Trust-building
9. `howto/delayed-rewards.rst` + `tests/test_howto_delayed_rewards.py`
10. `math/normal.rst`
11. `math/empirical-bayes.rst`
12. `math/policies.rst`
13. `howto/production.rst` + `tests/test_howto_production.py`

### Tier 3: Completeness
14. `math/normal-inverse-gamma.rst`, `math/intercept-only.rst`, `math/glm.rst`
15. `howto/sparse.rst` + `tests/test_howto_sparse.py`
16. Remove cut notebooks from repo, update cross-references
17. `changelog.rst`

## Verification

After each tier is implemented:
- `cd docs && make html` builds without warnings
- All internal cross-references resolve
- All how-to code snippet tests pass (`uv run pytest tests/test_howto_*.py`)
- Notebooks still render via nbsphinx
- API reference still generates correctly

## Infrastructure

Current setup (Sphinx + nbsphinx + autodoc + napoleon + pydata theme on ReadTheDocs) is fine. Migration to MkDocs is possible but the cost outweighs the benefit. `sphinx.ext.mathjax` added for math reference pages.

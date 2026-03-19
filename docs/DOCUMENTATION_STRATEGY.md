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
|   +-- howto/reward-functions.rst           DONE - Writing custom reward functions
|   |                                        (absorbs demo.ipynb content)
|   +-- howto/delayed-rewards.rst            DONE - Handling delayed rewards
|   +-- howto/production.rst                 DONE - Deploying a bandit to production
|   |                                        (absorbs persistence.ipynb content)
|   +-- howto/sparse.rst                     DONE - Working with sparse features / CHOLMOD
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
+-- Explanation
|   +-- explanation/worldview.rst               Bandits as forecasting under nonstationarity
|   +-- explanation/decision-theory.rst         Separating inference from decisions
|   +-- explanation/stabilized-forgetting.rst   Stabilized forgetting + empirical Bayes
|   +-- explanation/online-mackay.rst           Online evidence maximization
|   +-- explanation/exp3-bayesian.rst           EXP3 with Bayesian posteriors
|   +-- explanation/laplace-glm.rst             Laplace GLM in a conjugate framework
|   +-- explanation/eb-guard-rails.rst          EB numerical safeguards
|   +-- explanation/why-not-zooming.rst         Regression over zooming for continuous arms
|   +-- explanation/sparse-fill.rst             When sparse mode works (and when it doesn't)
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
- **Explanation pages** (`explanation/*.rst`): Why this library makes the mathematical choices it makes. The reasoning that connects the how-to guides ("do this") to the math reference ("here's the formula"). One page per topic so each can reference its own literature. See Explanation Specifications below.

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

### `howto/sparse.rst` -- Working with Sparse Features (DONE)
- When to enable `sparse=True`
- CHOLMOD vs. SuperLU backends (why SuperLU is slower: can't exploit SPD structure)
- Precision matrix fill: hierarchical features keep fill bounded, bag-of-words doesn't
- Code: sparse context matrix with 10k features
- Tests: `tests/test_howto_sparse.py` (4 tests)

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

## Explanation Specifications

Each page is prose-heavy, no runnable code. States the decision, the reasoning, references relevant literature, and is honest about what theoretical guarantees (if any) we have.

### `explanation/worldview.rst` -- Bandits as Forecasting Under Nonstationarity

The foundational page. Bandits in the real world are fundamentally forecasting problems — decisions happen in time, the world changes, and you need to act now with what you know. This library interprets bandits through that lens: anytime, contextual, nonstationary.

Touch on:
- The pull/update/decay loop is a Kalman filter with a decision layer: conjugate updates are the measurement update (incorporate new data for the arm you observed), decay is the prediction step (inject process noise — increase uncertainty on *all* arms every time step, whether observed or not)
- Decaying all arms isn't a design quirk — it's what the prediction step of a Kalman filter does. Unobserved arms still experience time passing. Their parameters may have drifted, so uncertainty should grow.
- Why conjugate priors: they give you closed-form Kalman-style updates. No MCMC, no variational inference, no mini-batches. One matrix operation per observation. This is what makes the library fast enough for real-time decisioning.
- The Bayesian framing unifies exploration and exploitation naturally (Thompson sampling is just acting on your beliefs) and gives you calibrated uncertainty for free
- Contrast with the textbook view: most bandit theory assumes stationary reward distributions and analyzes regret over a fixed horizon. This library assumes the world is always moving and optimizes for anytime performance.

### `explanation/decision-theory.rst` -- Separating Inference from Decisions

Bayesian decision theory separates two concerns: understanding the world (inference) and deciding how to act on that understanding (utility/loss). This library enforces that separation architecturally.

Touch on:
- The learner models *what actually happens* (CTR, conversion rate, revenue per click) — it maintains a posterior over the generative process
- The reward function is a utility function (negative loss) in the decision theory sense. It maps outcomes to *value* — what that outcome is worth to you (profit = revenue - cost, or any arbitrary utility). The library uses "reward function" following bandit convention, but this is the standard Bayesian decision-theoretic separation of posterior from loss.
- `update()` trains on raw outcomes, not transformed rewards. The posterior stays valid even if your utility function changes (e.g., ad costs change, you redefine success metrics). No retraining needed.
- The policy (Thompson sampling, UCB, etc.) is a third, separate concern: given the posterior and the utility, *how* do you select an action? This is the decision rule — minimizing expected loss (or equivalently maximizing expected utility/reward).
- Three orthogonal axes: learner (what do I believe?), reward/utility function (what do I value?), policy (how do I decide?). Changing one doesn't require changing the others.
- Reference: Berger (1985) *Statistical Decision Theory and Bayesian Analysis*, DeGroot (1970) *Optimal Statistical Decisions*

### `explanation/stabilized-forgetting.rst` -- Stabilized Forgetting + Empirical Bayes

Exponential forgetting (Kulhavy & Zarrop 1993) scales precision by γ^n, which drives the prior contribution toward zero over time — destroying regularization. Stabilized forgetting re-injects (1 - γ^n)α into the precision diagonal so the prior converges to α instead of vanishing. But we're *also* running MacKay EB to learn α. So the stabilization target is itself being updated by the data. Explain why this interaction is safe (EB updates α slowly, stabilization prevents catastrophic prior collapse between updates) and what would go wrong without either piece.

### `explanation/online-mackay.rst` -- Online Evidence Maximization

Textbook MacKay iterates to convergence on a fixed dataset. We do single-step MacKay updates per `partial_fit` call using exact decayed sufficient statistics recovered from the precision matrix. This is not an approximation — given Λ and the accumulated stats, the MacKay update equations are evaluated exactly. The "online" part is the choice to do one step per data arrival, which is natural because new data is imminent. Explain how sufficient statistics are maintained under decay, how X^TX is recovered from the precision matrix, and why this constitutes an exact online extension of MacKay's method.

### `explanation/exp3-bayesian.rst` -- EXP3 with Bayesian Posteriors

EXP3's core contribution is importance weighting to correct for selection-dependent observation — you don't pull arms uniformly, so your reward estimates are biased by your own policy. In an intercept-only model with no decay, the Bayesian posterior mean is just a weighted average of observed rewards — equivalent to EXP3's normalized reward sums. So the Bayesian version isn't a departure from EXP3; it's a natural generalization that reduces to the same thing in the simple case.

Touch on:
- Equivalence in the intercept-only case: posterior means ≈ normalized reward sums, importance weighting applies the same correction
- What the generalization buys: regression (condition on context, which standard EXP3 can't do), decay (natural nonstationarity handling), and full posterior uncertainty
- Importance weighting as correction for non-ignorable missingness: you observe rewards conditional on your selection policy, and conjugate posteriors can't correct for this on their own
- In adversarial environments: there is no true reward distribution, so the posterior is misspecified — but it still produces useful arm rankings, and the EXP3 selection machinery provides empirical robustness
- Honest about guarantees: EXP3's regret bound assumes its specific reward tracking, so we lose the formal bound in the extended case, but the debiasing principle still holds

### `explanation/laplace-glm.rst` -- Laplace GLM in a Conjugate Framework

The rest of the library is exact conjugate updates. The GLM uses IRLS for MAP estimation + Laplace approximation for the posterior. Explain why GLMs are worth the approximation cost (flexible likelihoods — logistic, Poisson) and what you give up (posterior is Gaussian at the MAP, not exact). The single-step IRLS option (n_iter=1) is a deliberate trade-off: if the previous posterior is a good initialization, one Newton step is usually sufficient for online updates. Explain when this is and isn't safe.

### `explanation/eb-guard-rails.rst` -- EB Numerical Safeguards

When β/α > 10^10, EB updates are silently rejected. This prevents numerical collapse in the underdetermined regime (p >> n), where MacKay can produce extreme hyperparameter ratios. Explain the failure mode (near-singular precision, garbage posterior), why a hard cutoff is preferable to a soft constraint, and what the user should know (EB is frozen until enough data arrives to make the problem well-determined).

### `explanation/why-not-zooming.rst` -- Regression Over Zooming for Continuous Arms

Lipschitz/continuous-armed bandits classically use zooming or adaptive discretization algorithms that partition the arm space and refine where rewards are promising. This library uses Bayesian regression over the arm space instead. Explain why.

Touch on:
- Zooming works well in the non-contextual, low-dimensional arm space — partition arm space, refine promising regions
- Adding context makes zooming intractable: you'd need to partition the joint (arm × context) space, which is combinatorially explosive in any realistic dimension
- Bayesian regression sidesteps this entirely — the design matrix encodes arm features and context jointly, and the posterior naturally provides adaptive resolution (tight where you have data, wide where you don't)
- Thompson sampling on the regression posterior gives you exploration for free — no explicit partitioning or confidence-radius bookkeeping needed
- Trade-off: you assume a parametric relationship (linear or GLM) between (arm, context) and reward, whereas zooming is nonparametric. In practice the parametric assumption is usually fine and buys you generalization across similar arms/contexts.

### `explanation/sparse-fill.rst` -- When Sparse Mode Works (and When It Doesn't)

The precision matrix update adds `x^T x` on each observation. Whether the precision matrix stays sparse over time depends on the feature structure.

Touch on:
- Each observation contributes nonzeros wherever its active features cross. With d_active active features per observation, that's up to d_active^2 new entries.
- Hierarchical features (country > state > city, one-hot at each level): each observation activates exactly one node per level, so the fill pattern is bounded and predictable. The precision matrix stays sparse indefinitely. This is the ideal use case.
- Arbitrary sparse features (e.g., bag-of-words): different observations activate different feature subsets. Over time, cross-feature entries accumulate and the precision matrix fills in. Eventually the sparse advantage erodes.
- The practical question: does `nnz(precision)` grow without bound, or does it plateau? Hierarchical features plateau. Bag-of-words doesn't.
- Cross-reference to `howto/sparse` for the mechanical setup.

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
10. ~~Docstring audit (policy See Also + Examples)~~ (DONE)
11. ~~`howto/reward-functions.rst` + `tests/test_howto_reward_functions.py`~~ (DONE)

### Tier 2: Trust-building
9. `explanation/*.rst` (8 explanation pages — no test files, prose only)
10. ~~`howto/delayed-rewards.rst` + `tests/test_howto_delayed_rewards.py`~~ (DONE)
11. `math/normal.rst`
12. `math/empirical-bayes.rst`
13. `math/policies.rst`
14. ~~`howto/production.rst` + `tests/test_howto_production.py`~~ (DONE)

### Tier 3: Completeness
14. `math/normal-inverse-gamma.rst`, `math/intercept-only.rst`, `math/glm.rst`
15. ~~`howto/sparse.rst` + `tests/test_howto_sparse.py`~~ (DONE)
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

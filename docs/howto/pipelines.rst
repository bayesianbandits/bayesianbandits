Integrating with sklearn Transformers
=====================================

``bayesianbandits`` estimators expect numeric arrays, but real data
arrives as JSON dicts, DataFrames, or raw features that need scaling.
Two wrapper classes let you plug in any sklearn transformer:

:class:`~bayesianbandits.AgentPipeline`
    Wraps an ``Agent`` or ``ContextualAgent``.  Transforms run once
    **before** every ``pull()`` and ``update()`` call.

:class:`~bayesianbandits.LearnerPipeline`
    Wraps the shared learner inside a ``LipschitzContextualAgent``.
    Transforms run **after** the arm featurizer enriches the context,
    right before the underlying estimator sees the data.

.. important::

   Every transformer must be **stateless** (like
   ``FunctionTransformer``) or **pre-fitted** before the pipeline is
   created.  Pipelines never call ``fit()`` on transformers.


Accept JSON input
------------------

Use ``DictVectorizer`` to convert dicts to sparse feature matrices:

.. code-block:: python

   from sklearn.feature_extraction import DictVectorizer
   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling, AgentPipeline,
   )

   vectorizer = DictVectorizer(sparse=True)
   vectorizer.fit([
       {"user_age": 25, "region": "US"},
       {"user_age": 40, "region": "EU"},
   ])

   arms = [
       Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
   pipeline = AgentPipeline(
       steps=[("vectorize", vectorizer)],
       final_agent=agent,
   )

   contexts = [{"user_age": 30, "region": "US"}]
   (action,) = pipeline.pull(contexts)
   pipeline.update(contexts, y=np.array([1.0]))


Accept DataFrame input
-----------------------

Use ``ColumnTransformer`` to handle mixed column types:

.. code-block:: python

   import pandas as pd
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling, AgentPipeline,
   )

   sample_df = pd.DataFrame({
       "age": [25, 40, 35],
       "region": ["US", "EU", "US"],
   })
   ct = ColumnTransformer([
       ("num", StandardScaler(), ["age"]),
       ("cat", OneHotEncoder(sparse_output=False), ["region"]),
   ])
   ct.fit(sample_df)

   arms = [
       Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
   pipeline = AgentPipeline(
       steps=[("preprocess", ct)],
       final_agent=agent,
   )

   df = pd.DataFrame({"age": [30], "region": ["US"]})
   (action,) = pipeline.pull(df)
   pipeline.update(df, y=np.array([1.0]))


Scale numeric features
-----------------------

Fit the scaler on a representative historical dataset. In an online
setting, any single batch may cover only a narrow slice of the input
range, so fitting on live data would produce unstable statistics:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling, AgentPipeline,
   )

   scaler = StandardScaler()
   scaler.fit(historical_features)  # shape (n_samples, n_features)

   arms = [
       Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
   pipeline = AgentPipeline(
       steps=[("scale", scaler)],
       final_agent=agent,
   )


Preprocess enriched features in a shared-learner bandit
--------------------------------------------------------

With :class:`~bayesianbandits.LipschitzContextualAgent`, the arm
featurizer adds an arm identity column to the context before the
shared learner sees it. Wrap the shared learner in a
:class:`~bayesianbandits.LearnerPipeline` so a ``ColumnTransformer``
can handle the mixed-type enriched DataFrame (numeric context + string
arm column).

**One-hot arm encoding (disjoint model):**

Each arm gets fully independent coefficients:

.. code-block:: python

   import pandas as pd
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from bayesianbandits import (
       Arm, LipschitzContextualAgent, NormalRegressor,
       ThompsonSampling, ArmColumnFeaturizer, LearnerPipeline,
   )

   arm_tokens = ["product_0", "product_1", "product_2"]

   # Pre-fit on representative enriched data (context + arm column)
   sample_enriched = pd.DataFrame({
       "age": [25, 40, 35] * 3,
       "score": [0.8, 0.6, 0.9] * 3,
       "arm": arm_tokens * 3,
   })
   ct = ColumnTransformer([
       ("num", StandardScaler(), ["age", "score"]),
       ("arm", OneHotEncoder(sparse_output=False), ["arm"]),
   ])
   ct.fit(sample_enriched)

   shared_learner = LearnerPipeline(
       steps=[("preprocess", ct)],
       learner=NormalRegressor(alpha=1.0, beta=1.0),
   )

   arms = [Arm(token) for token in arm_tokens]
   agent = LipschitzContextualAgent(
       arms=arms,
       policy=ThompsonSampling(),
       arm_featurizer=ArmColumnFeaturizer("arm"),
       learner=shared_learner,
   )

   context = pd.DataFrame({"age": [30], "score": [0.7]})
   (action,) = agent.pull(context)

**Shared context coefficients + per-arm offsets (hybrid model):**

Pass the context columns through unscaled alongside the one-hot arms.
The context coefficients are shared across all arms; the one-hot
columns act as per-arm intercept offsets. A new arm immediately
inherits the shared coefficients and only needs to learn its offset:

.. code-block:: python

   ct = ColumnTransformer([
       ("num", "passthrough", ["age", "score"]),
       ("arm", OneHotEncoder(sparse_output=False), ["arm"]),
   ])
   ct.fit(sample_enriched)

   # Design matrix: [age, score, is_product_0, is_product_1, is_product_2]
   shared_learner = LearnerPipeline(
       steps=[("preprocess", ct)],
       learner=NormalRegressor(alpha=1.0, beta=1.0),
   )

See the :doc:`hybrid bandits example </notebooks/hybrid-bandits>` for
a full comparison of disjoint vs. hybrid models with regret curves.


Access pipeline internals
--------------------------

Both pipeline types support indexing by name or position:

.. code-block:: python

   pipeline["vectorize"]              # by name
   name, transformer = pipeline[0]    # by position
   pipeline.named_steps               # {"vectorize": DictVectorizer(...)}

All agent methods (``add_arm``, ``remove_arm``, ``select_for_update``,
etc.) are forwarded through ``AgentPipeline`` unchanged.

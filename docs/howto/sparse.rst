Working with Sparse Features
============================

When your feature space is large and sparse (one-hot encoded
categories, text features, high-cardinality IDs), dense precision
matrices become impractical. A model with 100k features needs a
100k x 100k matrix in the dense case. Set ``sparse=True`` on the
estimator and the precision matrix is stored as a sparse CSC array,
so storage and updates scale with the number of nonzero entries
instead of the square of the feature dimension.

All linear estimators support ``sparse=True``. The intercept-only
models (:class:`~bayesianbandits.DirichletClassifier`,
:class:`~bayesianbandits.GammaRegressor`) don't need it.


Enable sparse mode
-------------------

Pass ``sparse=True`` to the estimator and provide context as
``scipy.sparse.csc_array``:

.. code-block:: python

   import numpy as np
   from scipy.sparse import random as sparse_random
   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling,
   )

   arms = [
       Arm(f"variant_{i}", learner=NormalRegressor(
           alpha=1.0, beta=1.0, sparse=True,
       ))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

   # Sparse context: 1 row, 10000 features, ~1% density
   X = sparse_random(1, 10000, density=0.01, format="csc", random_state=42)
   (action,) = agent.pull(X)
   agent.update(X, np.array([1.0]))

The estimator will not convert dense arrays to sparse for you. If you
pass a dense array to a ``sparse=True`` estimator, it will raise.

If the precision matrix fills in over time (common with unstructured
sparse features like bag-of-words), sparse operations become slower
than dense. Hierarchical features (one-hot at each level of a
taxonomy) keep fill bounded and are the ideal use case.


Install CHOLMOD for production workloads
-----------------------------------------

Two sparse backends are available:

**SuperLU** ships with scipy and works out of the box. It handles
arbitrary sparse LU decomposition, which is a harder problem than
what we actually need. Precision matrices are symmetric positive
definite, and SuperLU can't exploit that, so it does more work than
necessary. For small models this doesn't matter. For large ones it
dominates your inference time.

**CHOLMOD** (via ``scikit-sparse``) knows the matrix is symmetric
positive definite and takes advantage of it.

.. code-block:: bash

   pip install scikit-sparse

If ``scikit-sparse`` is installed, the library uses CHOLMOD
automatically. No code changes needed. Both backends apply
fill-reducing permutations internally; the library handles
unpermuting so that sampling, prediction, and all other operations
are unaware of the solver choice. To force SuperLU (for debugging or
benchmarking), set the environment variable ``BB_NO_SUITESPARSE=1``.

With CHOLMOD, real-time inference under 10 ms is feasible with models
up to 2\ :sup:`20` features.

from typing import Dict, Iterable, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class DirichletClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, alphas: Dict[Union[int, str], float], *, learning_rate: float = 1.0
    ):
        self.alphas = alphas
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X, y = check_X_y(X, y, copy=True, ensure_2d=True)
        self.classes_ = np.array([self.alphas.keys()])
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        if self.n_features_ > 1:
            raise NotImplementedError("Only one feature supported")

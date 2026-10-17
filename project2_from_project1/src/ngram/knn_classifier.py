"""A small cosine k-NN classifier for the n-gram feature matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Hashable

import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier


class NgramKNNClassifier:
    """Classify feature rows with uniform-vote cosine k-nearest neighbors.

    The default is ``k=3``. No adjustment is made when k exceeds the number of
    training rows. Equal vote counts follow scikit-learn's class ordering:
    the first class in ``classes_`` wins (labels are ordered by scikit-learn).
    Sparse inputs stay sparse; prediction returns an empty array for zero rows.
    """

    def __init__(self, n_neighbors: int = 3) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        self.n_neighbors = n_neighbors
        self._model: KNeighborsClassifier | None = None
        self._n_features: int | None = None

    def fit(
        self,
        X_train: Any,
        y_train: Sequence[Hashable] | np.ndarray | None,
    ) -> NgramKNNClassifier:
        """Fit on a 2-D feature matrix and one known label per training row."""
        X_train = self._as_matrix(X_train, "X_train")
        if X_train.shape[0] == 0:
            raise ValueError("X_train must contain at least one training row")
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature")
        if y_train is None:
            raise ValueError("y_train must contain training labels")

        labels = np.asarray(y_train)
        if labels.ndim != 1:
            raise ValueError("y_train must be a one-dimensional sequence")
        if labels.size != X_train.shape[0]:
            raise ValueError("X_train row count must match the number of y_train labels")
        if any(label is None for label in labels):
            raise ValueError("y_train cannot contain missing labels")
        if self.n_neighbors > X_train.shape[0]:
            raise ValueError(
                "n_neighbors cannot exceed the number of training rows "
                f"({X_train.shape[0]})"
            )

        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights="uniform",
            algorithm="brute",
            metric="cosine",
        )
        model.fit(X_train, labels)
        self._model = model
        self._n_features = X_train.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict one class per row; an empty 2-D input returns an empty array."""
        if self._model is None or self._n_features is None:
            raise RuntimeError("the k-NN classifier has not been fitted")

        X = self._as_matrix(X, "X")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X must have {self._n_features} features; got {X.shape[1]}"
            )
        if X.shape[0] == 0:
            return np.empty(0, dtype=self._model.classes_.dtype)
        return self._model.predict(X)

    @staticmethod
    def _as_matrix(matrix: Any, name: str) -> Any:
        if not issparse(matrix):
            matrix = np.asarray(matrix)
        shape = getattr(matrix, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        return matrix

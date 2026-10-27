"""Fixed kNN and query-centered weighted ridge local regression, in physical units."""
from __future__ import annotations

from . import runtime  # Configure caches/threads before numerical imports.

import numpy as np
from scipy.spatial import cKDTree


def inverse_distance_weights(distances: np.ndarray) -> tuple[np.ndarray, bool]:
    """Exact zeros: equal weight on the zero-distance neighbors, zero on others.

    Both models return that exact-match mean, avoiding an underdetermined local
    fit. With no exact matches, inverse Euclidean weights sum to one.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Distances must be a finite nonnegative vector")
    zero = distances == 0.0
    if zero.any():
        return zero.astype(np.float64) / zero.sum(), True
    # Scale by min distance first to avoid overflow; normalization is identical.
    weights = distances.min() / distances
    return weights / weights.sum(), False


def local_linear_intercept(offsets: np.ndarray, outputs: np.ndarray,
                           weights: np.ndarray, ridge: float) -> np.ndarray:
    """Minimize sum_i w_i ||a+B*offset_i-y_i||² + ridge*||B||².

    The intercept is unpenalized. Solve the augmented weighted system by SVD
    least squares, rather than forming an inverse or normal equations.
    """
    if ridge < 0:
        raise ValueError("Ridge must be nonnegative")
    offsets = np.asarray(offsets, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if offsets.ndim != 2 or outputs.ndim != 2 or len(offsets) != len(outputs) or weights.shape != (len(offsets),):
        raise ValueError("Incompatible local regression arrays")
    design = np.column_stack((np.ones(len(offsets)), offsets))
    sqrt_w = np.sqrt(weights)[:, None]
    lhs, rhs = design * sqrt_w, outputs * sqrt_w
    if ridge:
        penalty = np.zeros((offsets.shape[1], design.shape[1]))
        penalty[:, 1:] = np.sqrt(ridge) * np.eye(offsets.shape[1])
        lhs = np.vstack((lhs, penalty))
        rhs = np.vstack((rhs, np.zeros((offsets.shape[1], outputs.shape[1]))))
    coefficients, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coefficients[0]


class LocalProfileModels:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, k=32, ridge=1e-3):
        self.mean = np.mean(inputs, axis=0, dtype=np.float64)
        self.scale = np.std(inputs, axis=0, ddof=0, dtype=np.float64)
        if np.any(self.scale <= 0) or not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
            raise ValueError("Invalid training values or zero input variance")
        if not 1 <= k <= len(inputs):
            raise ValueError("Invalid neighbor count")
        self.inputs = (inputs - self.mean) / self.scale
        self.outputs = np.array(outputs, dtype=np.float64, copy=True)
        self.tree = cKDTree(self.inputs)
        self.k = k
        self.ridge = ridge

    def neighbors(self, queries: np.ndarray):
        standardized = (queries - self.mean) / self.scale
        distances, indices = self.tree.query(standardized, k=self.k, p=2, workers=1)
        if self.k == 1:
            distances, indices = distances[:, None], indices[:, None]
        # Fixed secondary ordering for returned equal-distance neighbors.
        for row in range(len(queries)):
            order = np.lexsort((indices[row], distances[row]))
            distances[row] = distances[row, order]
            indices[row] = indices[row, order]
        return standardized, distances, indices

    def predict(self, queries: np.ndarray, method: str) -> np.ndarray:
        if method not in ("knn", "local_linear"):
            raise ValueError(method)
        queries_std, distances, indices = self.neighbors(queries)
        predictions = np.empty((len(queries), self.outputs.shape[1]), dtype=np.float64)
        for i, (dist, neighbors) in enumerate(zip(distances, indices)):
            weights, exact = inverse_distance_weights(dist)
            neighbor_outputs = self.outputs[neighbors]
            if method == "knn" or exact:
                predictions[i] = weights @ neighbor_outputs
            else:
                offsets = self.inputs[neighbors] - queries_std[i]
                predictions[i] = local_linear_intercept(offsets, neighbor_outputs, weights, self.ridge)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Nonfinite model predictions")
        return predictions


def numerical_checks() -> dict:
    """Tiny synthetic code checks only; these are not benchmark evidence."""
    offsets = (np.vstack((np.eye(4), -np.eye(4), np.ones((1, 4)), -np.ones((1, 4))))
               + np.array([0.2, -0.1, 0.35, 0.15]))
    intercept = np.array([2.5, -3.0, 12.0])
    slopes = np.arange(12, dtype=np.float64).reshape(4, 3) / 7
    outputs = intercept + offsets @ slopes
    distances = np.linalg.norm(offsets, axis=1)
    weights, _ = inverse_distance_weights(distances)
    recovered = local_linear_intercept(offsets, outputs, weights, ridge=0)
    np.testing.assert_allclose(recovered, intercept, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(weights @ outputs - intercept)) > 0.01, "Affine check must distinguish regression from simple averaging"
    # Large slope penalty must still preserve a constant intercept.
    constant = np.broadcast_to(intercept, outputs.shape)
    np.testing.assert_allclose(local_linear_intercept(offsets, constant, weights, ridge=1000), intercept, rtol=1e-12, atol=1e-12)
    w_zero, exact = inverse_distance_weights(np.array([0., 0., 2.]))
    np.testing.assert_array_equal(w_zero, [0.5, 0.5, 0.0])
    assert exact
    # Exercise both full prediction paths with duplicate exact training inputs.
    x = np.vstack((np.zeros((2, 4)), np.eye(4), -np.eye(4)))
    y = np.column_stack((np.arange(len(x)), np.arange(len(x)) ** 2)).astype(float)
    model = LocalProfileModels(x, y, k=len(x))
    for method in ("knn", "local_linear"):
        np.testing.assert_allclose(model.predict(x[:1], method), y[:2].mean(axis=0, keepdims=True), rtol=0, atol=1e-15)
    return {"passed": True, "evidence_type": "synthetic numerical code check, not benchmark evidence",
            "affine_intercept_max_absolute_error": float(np.max(np.abs(recovered - intercept))),
            "weighted_design_condition_number": float(np.linalg.cond(np.column_stack((np.ones(len(offsets)), offsets)) * np.sqrt(weights[:, None]))),
            "affine_ridge": 0, "constant_intercept_with_large_ridge": "passed",
            "duplicate_zero_distance_paths_for_both_methods": "passed"}

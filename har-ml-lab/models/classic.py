from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


@dataclass
class LogisticRegressionModel:
    """
    Simple wrapper around sklearn's LogisticRegression
    with a clean fit/predict interface.
    """

    # C: Inverse of regularization strength. Smaller values specify stronger regularization.
    C: float = 1.0
    # max_iter: Maximum number of iterations taken for the solvers to converge.
    max_iter: int = 1000
    # solver: Algorithm to use in the optimization problem.
    solver: str = "lbfgs"
    # n_jobs: Number of CPU cores used when parallelizing.
    n_jobs: int = 1

    # Internal variable to store the actual sklearn model
    _model: Optional[LogisticRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Train the model."""
        # Prepare kwargs for LogisticRegression
        kwargs = {
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "n_jobs": self.n_jobs,
        }
        
        # Check if 'multi_class' is supported (for older sklearn versions)
        import inspect
        sig = inspect.signature(LogisticRegression)
        if "multi_class" in sig.parameters:
            kwargs["multi_class"] = "auto"
            
        self._model = LogisticRegression(**kwargs)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")
        return self._model.predict(X)


@dataclass
class SVMModel:
    """
    Wrapper for Support Vector Machine (SVM) Classifier.
    
    SVM is effective in high dimensional spaces.
    """

    # C: Regularization parameter. The strength of the regularization is inversely proportional to C.
    # Must be strictly positive.
    C: float = 1.0
    
    # kernel: Specifies the kernel type to be used in the algorithm.
    # 'rbf' is usually a good default.
    kernel: str = "rbf"
    
    # gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    # 'scale' uses 1 / (n_features * X.var()) as value of gamma.
    gamma: str | float = "scale"

    _model: Optional[SVC] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        """
        Train the SVM model.
        Note: SVM training can be slow on large datasets.
        """
        self._model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=42  # Fixed random state for reproducibility
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict(X)


@dataclass
class RandomForestModel:
    """
    Wrapper for Random Forest Classifier.
    
    A random forest is a meta estimator that fits a number of decision tree classifiers.
    """
    
    # n_estimators: The number of trees in the forest.
    n_estimators: int = 100
    
    # max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
    max_depth: Optional[int] = None
    
    # min_samples_split: The minimum number of samples required to split an internal node.
    min_samples_split: int = 2
    
    # n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    n_jobs: int = -1
    
    _model: Optional[RandomForestClassifier] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the Random Forest model."""
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_jobs=self.n_jobs,
            random_state=42
        )
        self._model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict(X)

"""
PLS (Partial Least Squares) model with cross-validation implementation.
This module provides a class-based implementation of PLS regression with integrated
cross-validation capabilities using scikit-learn.
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Any, Optional, Union, cast, TypeVar
from src.dataset import Dataset
from src.utils import _detect_numerical_features
import pandas as pd

# Type variable for numpy arrays
ArrayType = TypeVar("ArrayType", bound=np.ndarray)


class PLSModel(BaseEstimator, RegressorMixin):
    """
    A PLS regression model with cross-validation capabilities.
    The optimal number of components is determined through cross-validation.
    Handles categorical variables automatically using OneHotEncoder.

    Parameters
    ----------
    max_components : int, optional (default=10)
        Maximum number of components to test
    cv_folds : int, optional (default=5)
        Number of folds for cross-validation
    scoring : str, optional (default='r2')
        Scoring metric for cross-validation

    Attributes
    ----------
    model_ : PLSRegression
        The fitted PLS model with optimal number of components
    cv_results_ : dict
        Results from cross-validation for each number of components
    best_score_ : float
        Best cross-validation score
    n_components_ : int
        Optimal number of components found by cross-validation
    encoder_ : OneHotEncoder
        OneHotEncoder for categorical features
    numerical_features_ : ndarray
        Array of indices of numerical features
    categorical_features_ : ndarray
        Array of indices of categorical features (deduced from numerical_features_)
    target_types_ : dict[str, str]
        Dictionary mapping target names to their types ('regression' or 'classification')
    target_categories_ : dict[str, list[str]]
        Dictionary to store original categories for each classification target
    """

    def __init__(
        self,
        max_components: int = 10,
        cv_folds: int = 5,
        scoring: str = "r2",
    ):
        self.max_components = max_components
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.model_: Optional[PLSRegression] = None
        self.cv_results_: Dict = {}
        self.best_score_: Optional[float] = None
        self.n_components_: Optional[int] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.numerical_features_: Optional[NDArray[np.int_]] = None
        self.categorical_features_: Optional[NDArray[np.int_]] = None
        self.target_types_: dict[str, str] = {}
        self.target_categories_: dict[str, list[str]] = {}

    def _prepare_features(self, dataset: Dataset) -> NDArray[np.float64]:
        """
        Prepare features by detecting numerical features and applying OneHotEncoder to categorical variables.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing both input features and target variables

        Returns
        -------
        X : np.ndarray
            Processed features with categorical variables encoded
        """
        X = dataset.get_input_data()

        if self.encoder_ is None:
            # Detect numerical features
            self.numerical_features_ = _detect_numerical_features(X)
            # Deduce categorical features
            self.categorical_features_ = np.array(
                [i for i in range(X.shape[1]) if i not in self.numerical_features_]
            )

            # Initialize encoder if there are categorical features
            if self.categorical_features_.size > 0:
                self.encoder_ = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                self.encoder_.fit(X.iloc[:, self.categorical_features_])

        # Process features
        if (
            self.numerical_features_ is not None
            and self.categorical_features_ is not None
            and self.encoder_ is not None
            and self.categorical_features_.size > 0
        ):
            X_num = X.iloc[:, self.numerical_features_].astype(np.float64)
            X_cat = self.encoder_.transform(X.iloc[:, self.categorical_features_])
            # Ensure both arrays are 2D
            if X_num.ndim == 1:
                X_num = X_num.to_numpy().reshape(-1, 1)
            else:
                X_num = X_num.to_numpy()
            # Convert sparse matrix to dense array
            X_cat = np.asarray(X_cat)
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
            return cast(NDArray[np.float64], np.hstack((X_num, X_cat)))

        return X.astype(np.float64).to_numpy()

    def _cross_validate_components(self, X: NDArray, y: NDArray) -> tuple[int, float]:
        """
        Perform cross-validation for different numbers of components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        best_n_components : int
            Optimal number of components
        best_score : float
            Best cross-validation score
        """
        best_score = -np.inf
        best_n_components = 1

        # Test different numbers of components
        for n_comp in range(1, min(self.max_components + 1, X.shape[1] + 1)):
            model = PLSRegression(n_components=n_comp)
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=self.cv_folds,
                scoring=self.scoring,
                return_train_score=True,
            )

            mean_test_score = np.mean(cv_results["test_score"])
            self.cv_results_[n_comp] = cv_results

            # Update best score if current is better
            if mean_test_score > best_score:
                best_score = mean_test_score
                best_n_components = n_comp

        return best_n_components, best_score

    def _detect_task_types(self, y_df: pd.DataFrame) -> tuple[NDArray, dict[str, str]]:
        """
        Detect whether each target variable requires regression or classification
        and prepare the target variables accordingly.

        Parameters
        ----------
        y_df : pd.DataFrame
            DataFrame containing the target variables

        Returns
        -------
        tuple[NDArray, dict[str, str]]
            - Processed target variables as numpy array
            - Dictionary mapping target names to their types ('regression' or 'classification')
        """
        # Detect numerical features in target data
        target_numerical_features = _detect_numerical_features(y_df)
        target_types = {}
        y = y_df.copy()

        for i, col in enumerate(y_df.columns):
            if i in target_numerical_features:
                target_types[col] = "regression"
                y[col] = y[col].astype(np.float64)
            else:
                target_types[col] = "classification"
                # Store original categories before converting to codes
                categories = pd.Categorical(y[col])
                self.target_categories_[col] = categories.categories.tolist()
                y[col] = categories.codes

        return y.to_numpy(), target_types

    def fit(self, dataset: Dataset, target_labels: list[str]) -> "PLSModel":
        """
        Fit the PLS model after finding optimal number of components through cross-validation.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing both input features and target variables
        target_labels : list[str]
            Names of the target variables to predict

        Returns
        -------
        self : returns an instance of self

        Raises
        ------
        ValueError
            If model has already been fitted or if target_labels are invalid
        """
        if self.model_ is not None:
            raise ValueError(
                "Model has already been fitted. Create a new instance for retraining."
            )

        # Prepare features and get target data
        X = self._prepare_features(dataset)
        y_df = dataset.get_output_data(target_labels)
        y, self.target_types_ = self._detect_task_types(y_df)

        # Find optimal number of components
        self.n_components_, self.best_score_ = self._cross_validate_components(X, y)

        # Fit final model with optimal number of components
        self.model_ = PLSRegression(n_components=self.n_components_)
        self.model_.fit(X, y)

        return self

    def predict(self, X: NDArray | Dataset) -> dict[str, NDArray]:
        """
        Predict using the PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or Dataset
            Samples or Dataset object

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping target names to their predictions, with categorical variables
            converted back to their original labels

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if isinstance(X, Dataset):
            X_processed = self._prepare_features(X)
        else:
            X_processed = X
            if (
                self.categorical_features_ is not None
                and self.categorical_features_.size > 0
                and self.encoder_ is not None
            ):
                numerical_features = np.array(
                    [
                        i
                        for i in range(X.shape[1])
                        if i not in self.categorical_features_
                    ],
                    dtype=np.int_,
                )
                X_num = X[:, numerical_features]
                X_cat = self.encoder_.transform(X[:, self.categorical_features_])
                X_cat = np.asarray(X_cat)
                X_processed = np.concatenate([X_num.astype(np.float64), X_cat.astype(np.float64)], axis=1)

        # Get raw predictions
        raw_predictions = self.model_.predict(X_processed)
        
        # Convert predictions to dictionary
        predictions = {}
        
        for i, (target, target_type) in enumerate(self.target_types_.items()):
            if target_type == "regression":
                predictions[target] = raw_predictions[:, i]
            else:  # classification
                # Round to nearest integer for classification
                pred_codes = np.round(raw_predictions[:, i]).astype(int)
                # Clip to ensure indices are within bounds
                pred_codes = np.clip(pred_codes, 0, len(self.target_categories_[target]) - 1)
                # Convert back to original categories
                predictions[target] = np.array([
                    self.target_categories_[target][code]
                    for code in pred_codes
                ])
        
        return predictions

    def transform(self, X: Union[NDArray, Dataset]) -> NDArray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or Dataset
            Samples or Dataset object

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            Transformed data

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if isinstance(X, Dataset):
            X_processed = self._prepare_features(X)
        else:
            X_processed = X
            if (
                self.categorical_features_ is not None
                and self.categorical_features_.size > 0
                and self.encoder_ is not None
            ):
                numerical_features = np.array(
                    [
                        i
                        for i in range(X.shape[1])
                        if i not in self.categorical_features_
                    ],
                    dtype=np.int_,
                )
                X_num = X[:, numerical_features]
                X_cat = self.encoder_.transform(X[:, self.categorical_features_])
                X_cat = np.asarray(X_cat)
                X_processed = np.concatenate([X_num.astype(np.float64), X_cat.astype(np.float64)], axis=1)

        return cast(NDArray, self.model_.transform(X_processed))

    def get_cv_results(self) -> Dict:
        """
        Get cross-validation results for all tested number of components.

        Returns
        -------
        dict
            Dictionary containing cross-validation results for each number of components

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.cv_results_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.cv_results_

    def plot_cv_results(self, figsize: tuple[int, int] = (10, 6)) -> None:
        """
        Plot cross-validation results showing train and test scores for different numbers of components.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Figure size (width, height) in inches, by default (10, 6)

        Raises
        ------
        ValueError
            If model has not been fitted
        ImportError
            If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. Please install it with: pip install matplotlib"
            )

        if self.cv_results_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        cv_results = self.get_cv_results()
        components = list(cv_results.keys())
        test_scores = [np.mean(cv_results[n_comp]["test_score"]) for n_comp in components]
        train_scores = [np.mean(cv_results[n_comp]["train_score"]) for n_comp in components]

        plt.figure(figsize=figsize)
        plt.plot(components, test_scores, "o-", label="Test Score")
        plt.plot(components, train_scores, "o-", label="Train Score")
        plt.xlabel("Number of Components")
        plt.ylabel("R² Score")
        plt.title("PLS Cross-validation Results")
        plt.legend()
        plt.grid(True)
        plt.show()

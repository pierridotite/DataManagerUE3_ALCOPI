"""
Random Forest model with cross-validation implementation.
This module provides a class-based implementation of Random Forest with integrated
cross-validation capabilities for hyperparameter tuning using scikit-learn.
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import cast, TYPE_CHECKING, Any

from src.dataset import Dataset
from src.utils import _detect_numerical_features

from sklearn.ensemble._forest import RandomForestClassifier as RandomForestClassifierType
from sklearn.multioutput import MultiOutputClassifier as MultiOutputClassifierType


class RandomForestModel(BaseEstimator):
    """
    A Random Forest model with cross-validation capabilities.
    The optimal hyperparameters are determined through grid search cross-validation.
    Automatically handles both regression and classification tasks based on data types.

    Parameters
    ----------
    param_grid : dict[str, list[Any] | Any] | None, optional
        Dictionary with parameters names (string) as keys and lists of parameter settings
        or single values to try as values. If a single value is provided, it will be used
        directly without cross-validation for this parameter.
        Default is:
        {
            'n_estimators': [100],
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 2
        }
    cv_folds : int, optional (default=5)
        Number of folds for cross-validation

    Attributes
    ----------
    models_ : dict[str, RandomForestRegressor | RandomForestClassifier | MultiOutputClassifier]
        Dictionary of fitted models for each target type
    cv_results_ : dict[str, dict]
        Results from cross-validation for different hyperparameter combinations
    best_scores_ : dict[str, float]
        Best cross-validation scores for each target
    best_params_ : dict[str, dict]
        Best hyperparameters found by cross-validation for each target
    target_types_ : dict[str, str]
        Dictionary mapping target names to their types ('regression' or 'classification')
    numerical_features_ : NDArray[np.int_] | None
        Array of indices of numerical features in input data
    categorical_features_ : NDArray[np.int_] | None
        Array of indices of categorical features in input data
    target_categories_ : dict[str, list[str]]
        Dictionary to store original categories for each classification target
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any] | Any] | None = None,
        cv_folds: int = 5,
    ):
        # Set default param_grid
        default_param_grid = {
            "n_estimators": [100],
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
        }

        # Initialize param_grid with defaults, then update with provided values
        self.param_grid = {}
        param_grid = param_grid or {}

        # For each parameter in default grid, use provided value if exists, otherwise use default
        for param, default_value in default_param_grid.items():
            value = param_grid.get(param, default_value)
            self.param_grid[param] = [value] if not isinstance(value, list) else value

        # Add any additional parameters from param_grid that weren't in defaults
        for param, value in param_grid.items():
            if param not in default_param_grid:
                self.param_grid[param] = (
                    [value] if not isinstance(value, list) else value
                )

        self.cv_folds = cv_folds

        self.models_: dict[
            str,
            RandomForestRegressor | RandomForestClassifier | MultiOutputClassifier,
        ] = {}
        self.cv_results_: dict[str, dict] = {}
        self.best_scores_: dict[str, float] = {}
        self.best_params_: dict[str, dict] = {}
        self.target_types_: dict[str, str] = {}
        self.numerical_features_: NDArray[np.int_] | None = None
        self.categorical_features_: NDArray[np.int_] | None = None
        self.encoder_: OneHotEncoder | None = None
        self.target_categories_: dict[str, list[str]] = {}  # Store original categories for each classification target

    def prepare_features(self, dataset: Dataset) -> NDArray:
        """
        Prepare features by detecting numerical features and applying OneHotEncoder to categorical variables.
        This method only handles input features, not target variables.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the input features

        Returns
        -------
        NDArray
            Processed input features as numpy array
        """
        # Get input data as pandas DataFrame
        X = dataset.get_input_data()

        if self.numerical_features_ is None:
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

            # Convert sparse matrix to dense array if needed
            X_cat = np.asarray(X_cat)
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)

            return cast(NDArray[np.float64], np.hstack((X_num, X_cat)))

        return X.astype(np.float64).to_numpy()

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

    def fit(self, dataset: Dataset, target_labels: list[str]) -> "RandomForestModel":
        """
        Fit Random Forest models after finding optimal hyperparameters through cross-validation.
        Automatically detects and handles both regression and classification tasks.

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
        if self.models_:
            raise ValueError(
                "Model has already been fitted. Create a new instance for retraining."
            )
        # Prepare input features
        X = self.prepare_features(dataset)

        # Get and prepare target variables
        y_df = dataset.get_output_data(target_labels)
        y, self.target_types_ = self._detect_task_types(y_df)

        # Separate regression and classification targets
        regression_targets = [
            i
            for i, (col, type_) in enumerate(self.target_types_.items())
            if type_ == "regression"
        ]
        classification_targets = [
            i
            for i, (col, type_) in enumerate(self.target_types_.items())
            if type_ == "classification"
        ]

        # Handle regression targets
        if regression_targets:
            y_reg = (
                y[:, regression_targets]
                if len(regression_targets) > 1
                else y[:, regression_targets[0]]
            )

            # Initialize base model with single values from param_grid
            base_params = {k: v[0] for k, v in self.param_grid.items()}
            base_model = RandomForestRegressor(random_state=42, **base_params)

            grid_search = GridSearchCV(
                base_model, self.param_grid, cv=self.cv_folds, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X, y_reg)

            self.cv_results_["regression"] = grid_search.cv_results_
            self.best_scores_["regression"] = grid_search.best_score_
            self.best_params_["regression"] = grid_search.best_params_

            # Fit final regression model
            self.models_["regression"] = RandomForestRegressor(
                **self.best_params_["regression"], random_state=42
            )
            self.models_["regression"].fit(X, y_reg)

        # Handle classification targets
        if classification_targets:
            y_clf = (
                y[:, classification_targets]
                if len(classification_targets) > 1
                else y[:, classification_targets[0]]
            )

            if len(classification_targets) == 1:
                # Single classification target
                # Initialize base model with single values from param_grid
                base_params = {k: v[0] for k, v in self.param_grid.items()}
                base_model = RandomForestClassifier(random_state=42, **base_params)

                grid_search = GridSearchCV(
                    base_model,
                    self.param_grid,
                    cv=self.cv_folds,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                grid_search.fit(X, y_clf.ravel())

                self.cv_results_["classification"] = grid_search.cv_results_
                self.best_scores_["classification"] = grid_search.best_score_
                self.best_params_["classification"] = grid_search.best_params_

                # Fit final classification model
                self.models_["classification"] = RandomForestClassifier(
                    **self.best_params_["classification"], random_state=42
                )
                self.models_["classification"].fit(X, y_clf.ravel())
            else:
                # Multiple classification targets
                # Initialize base model with single values from param_grid
                base_params = {k: v[0] for k, v in self.param_grid.items()}
                base_model = MultiOutputClassifier(
                    RandomForestClassifier(random_state=42, **base_params)
                )

                # Adapt param_grid for MultiOutputClassifier
                multi_param_grid = {
                    f"estimator__{k}": v for k, v in self.param_grid.items()
                }

                grid_search = GridSearchCV(
                    base_model,
                    multi_param_grid,
                    cv=self.cv_folds,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                grid_search.fit(X, y_clf)

                self.cv_results_["multi_classification"] = grid_search.cv_results_
                self.best_scores_["multi_classification"] = grid_search.best_score_
                self.best_params_["multi_classification"] = grid_search.best_params_

                # Fit final multi-classification model
                self.models_["multi_classification"] = MultiOutputClassifier(
                    RandomForestClassifier(
                        **{
                            k.replace("estimator__", ""): v
                            for k, v in self.best_params_[
                                "multi_classification"
                            ].items()
                        },
                        random_state=42,
                    )
                )
                self.models_["multi_classification"].fit(X, y_clf)

        return self

    def predict(self, X: NDArray | Dataset) -> dict[str, NDArray]:
        """
        Predict using the Random Forest models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or Dataset
            Samples or Dataset object

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping target names to their predictions

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not self.models_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if isinstance(X, Dataset):
            X_processed = self.prepare_features(X)
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
                X_processed = np.concatenate(
                    [X_num.astype(np.float64), X_cat.astype(np.float64)], axis=1
                )

        predictions = {}

        # Get predictions for regression targets
        if "regression" in self.models_:
            regression_preds = np.asarray(self.models_["regression"].predict(X_processed))
            regression_targets = [
                col
                for col, type_ in self.target_types_.items()
                if type_ == "regression"
            ]
            if regression_preds.ndim == 1:
                predictions[regression_targets[0]] = regression_preds
            else:
                for i, target in enumerate(regression_targets):
                    predictions[target] = regression_preds[:, i]

        # Get predictions for classification targets
        if "classification" in self.models_:
            classification_preds = np.asarray(self.models_["classification"].predict(X_processed))
            classification_target = next(
                col
                for col, type_ in self.target_types_.items()
                if type_ == "classification"
            )
            # Convert numerical predictions back to original categories
            predictions[classification_target] = np.array([
                self.target_categories_[classification_target][int(pred)]
                for pred in classification_preds
            ])

        # Get predictions for multi-classification targets
        if "multi_classification" in self.models_:
            multi_clf_preds = np.asarray(self.models_["multi_classification"].predict(X_processed))
            multi_clf_targets = [
                col
                for col, type_ in self.target_types_.items()
                if type_ == "classification"
            ]
            for i, target in enumerate(multi_clf_targets):
                # Convert numerical predictions back to original categories
                predictions[target] = np.array([
                    self.target_categories_[target][int(pred)]
                    for pred in multi_clf_preds[:, i]
                ])

        return predictions

    def predict_proba(self, X: NDArray | Dataset) -> dict[str, NDArray]:
        """
        Predict class probabilities for classification targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or Dataset
            Samples or Dataset object

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping classification target names to their class probabilities

        Raises
        ------
        ValueError
            If model has not been fitted or if there are no classification targets
        """
        if not any(type_ == "classification" for type_ in self.target_types_.values()):
            raise ValueError("No classification targets available")

        if isinstance(X, Dataset):
            X_processed = self.prepare_features(X)
        else:
            X_processed = X

        probabilities = {}

        # Get probabilities for single classification target
        if "classification" in self.models_:
            classification_probs = cast(RandomForestClassifier, self.models_["classification"]).predict_proba(
                X_processed
            )
            classification_target = next(
                col
                for col, type_ in self.target_types_.items()
                if type_ == "classification"
            )
            probabilities[classification_target] = classification_probs

        # Get probabilities for multi-classification targets
        if "multi_classification" in self.models_:
            multi_clf_probs = cast(MultiOutputClassifier, self.models_["multi_classification"]).predict_proba(
                X_processed
            )
            multi_clf_targets = [
                col
                for col, type_ in self.target_types_.items()
                if type_ == "classification"
            ]
            for i, target in enumerate(multi_clf_targets):
                probabilities[target] = multi_clf_probs[i]

        return probabilities

    def get_cv_results(self) -> dict[str, dict]:
        """
        Get cross-validation results for all models.

        Returns
        -------
        dict[str, dict]
            Dictionary containing cross-validation results for each target type

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not self.cv_results_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.cv_results_

    def plot_cv_results(
        self,
        param_name: str,
        model_type: str = "regression",
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot cross-validation results for a specific parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to plot
        model_type : str, optional
            Type of model to plot results for ('regression', 'classification', or 'multi_classification')
        figsize : tuple[int, int], optional
            Figure size (width, height) in inches, by default (10, 6)

        Raises
        ------
        ValueError
            If model has not been fitted or if model_type is invalid
        ImportError
            If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. Please install it with: pip install matplotlib"
            )

        if not self.cv_results_ or model_type not in self.cv_results_:
            raise ValueError(f"No cross-validation results available for {model_type}")

        # Extract parameter values and corresponding scores
        cv_results = self.cv_results_[model_type]
        param_key = f"param_{param_name}"
        if param_key not in cv_results:
            raise ValueError(
                f"Parameter {param_name} not found in cross-validation results"
            )

        # Convert parameter values to list and handle None values
        param_values = cv_results[param_key]
        param_values = [str(v) if v is not None else "None" for v in param_values]
        test_scores = cv_results["mean_test_score"]
        train_scores = cv_results["mean_train_score"]

        plt.figure(figsize=figsize)
        plt.plot(param_values, test_scores, "o-", label="Test Score")
        plt.plot(param_values, train_scores, "o-", label="Train Score")
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.title(f"Random Forest {model_type.title()} - {param_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

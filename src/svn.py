"""
Support Vector Machine implementation with both LS-SVN (classification) and SVR (regression) capabilities.
This module provides a class-based implementation of SVM with integrated cross-validation 
capabilities using scikit-learn.
"""

from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import make_scorer, r2_score, accuracy_score
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any, cast
from src.dataset import Dataset
from src.utils import _detect_numerical_features
import pandas as pd


def _multi_target_r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Calculate R² score for multi-target regression.
    Returns the mean R² score across all targets.
    """
    scores = np.array(
        [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    )
    return float(np.mean(scores))


def _multi_target_accuracy_score(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Calculate accuracy score for multi-target classification.
    Returns the mean accuracy score across all targets.
    """
    scores = np.array(
        [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    )
    return float(np.mean(scores))


class SVNModel(BaseEstimator):
    """
    A Support Vector Machine model with both classification (LS-SVN) and regression (SVR) capabilities.
    The optimal hyperparameters are determined through grid search cross-validation.
    Handles categorical variables automatically using OneHotEncoder.
    Supports both multi-output and single-output approaches.

    Parameters
    ----------
    param_grid : dict[str, dict[str, list[Any]]] | None, optional
        Dictionary with parameters names (string) as keys and lists of parameter settings
        to try as values. Default parameters are provided if not specified.
    cv_folds : int, optional (default=5)
        Number of folds for cross-validation
    scoring : str, optional (default='accuracy' for classification, 'r2' for regression)
        Scoring metric for cross-validation
    use_multioutput : bool, optional (default=False)
        Whether to use MultiOutputClassifier/MultiOutputRegressor for multiple targets
        If False, will train separate models for each target

    Attributes
    ----------
    models_ : dict[str, SVC | SVR | MultiOutputClassifier | MultiOutputRegressor]
        Dictionary containing fitted models for each target type or target name
    cv_results_ : dict[str, dict]
        Results from cross-validation for each model type or target name
    best_params_ : dict[str, dict]
        Best parameters found by grid search for each model type or target name
    best_scores_ : dict[str, float]
        Best cross-validation scores for each model type or target name
    encoder_ : OneHotEncoder
        OneHotEncoder for categorical features
    scaler_ : StandardScaler
        StandardScaler for feature scaling
    numerical_features_ : ndarray
        Array of indices of numerical features
    categorical_features_ : ndarray
        Array of indices of categorical features
    target_types_ : dict[str, str]
        Dictionary mapping target names to their types ('regression' or 'classification')
    target_categories_ : dict[str, list[str]]
        Dictionary to store original categories for each classification target
    """

    def __init__(
        self,
        param_grid: dict[str, dict[str, list[Any]]] | None = None,
        cv_folds: int = 5,
        scoring: str | None = None,
        use_multioutput: bool = False,
    ):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_multioutput = use_multioutput

        # Default values for classification parameters
        default_c_values = [0.1, 1.0, 10.0, 100.0]
        default_kernel_values = ["linear", "rbf"]
        default_gamma_values = ["scale", "auto", 0.1, 1.0]

        # Default values for regression parameters
        default_epsilon_values = [0.1, 0.2, 0.3]

        # Set up classification parameter grid with fallback to defaults
        if param_grid and "classification" in param_grid:
            classification_params = param_grid["classification"]
            self.classification_param_grid = {
                "C": classification_params.get("C", default_c_values),
                "kernel": classification_params.get("kernel", default_kernel_values),
                "gamma": classification_params.get("gamma", default_gamma_values),
            }
        else:
            self.classification_param_grid = {
                "C": default_c_values,
                "kernel": default_kernel_values,
                "gamma": default_gamma_values,
            }

        # Set up regression parameter grid with fallback to defaults
        if param_grid and "regression" in param_grid:
            regression_params = param_grid["regression"]
            self.regression_param_grid = {
                "C": regression_params.get("C", default_c_values),
                "kernel": regression_params.get("kernel", default_kernel_values),
                "gamma": regression_params.get("gamma", default_gamma_values),
                "epsilon": regression_params.get("epsilon", default_epsilon_values),
            }
        else:
            self.regression_param_grid = {
                "C": default_c_values,
                "kernel": default_kernel_values,
                "gamma": default_gamma_values,
                "epsilon": default_epsilon_values,
            }

        # Initialize attributes
        self.models_: dict[
            str, SVC | SVR | MultiOutputClassifier | MultiOutputRegressor
        ] = {}
        self.cv_results_: dict[str, dict] = {}
        self.best_params_: dict[str, dict] = {}
        self.best_scores_: dict[str, float] = {}
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.numerical_features_: NDArray[np.int_] | None = None
        self.categorical_features_: NDArray[np.int_] | None = None
        self.target_types_: dict[str, str] = {}
        self.target_categories_: dict[str, list[str]] = {}

    def prepare_features(self, dataset: Dataset) -> NDArray[np.float64]:
        """
        Prepare features by detecting numerical features and applying preprocessing.
        """
        X = dataset.get_input_data()

        if self.encoder_ is None or self.scaler_ is None:
            # Detect numerical features
            self.numerical_features_ = _detect_numerical_features(X)
            # Deduce categorical features
            self.categorical_features_ = np.array(
                [i for i in range(X.shape[1]) if i not in self.numerical_features_]
            )

            # Initialize encoder if there are categorical features
            if len(self.categorical_features_) > 0:
                self.encoder_ = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                self.encoder_.fit(X.iloc[:, self.categorical_features_])

            # Initialize scaler for numerical features
            self.scaler_ = StandardScaler()
            if len(self.numerical_features_) > 0:
                self.scaler_.fit(X.iloc[:, self.numerical_features_])

        # Process features
        X_processed: NDArray[np.float64] = np.zeros((X.shape[0], 0), dtype=np.float64)

        # Scale numerical features
        if self.numerical_features_ is not None and len(self.numerical_features_) > 0:
            X_num = np.asarray(
                self.scaler_.transform(X.iloc[:, self.numerical_features_]),
                dtype=np.float64,
            )
            X_processed = X_num

        # Encode categorical features
        if (
            self.categorical_features_ is not None
            and len(self.categorical_features_) > 0
            and self.encoder_ is not None
        ):
            X_cat = np.asarray(
                self.encoder_.transform(X.iloc[:, self.categorical_features_]),
                dtype=np.float64,
            )
            if X_processed.shape[1] > 0:
                X_processed = np.concatenate((X_processed, X_cat), axis=1)
            else:
                X_processed = X_cat

        return X_processed

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

    def fit(self, dataset: Dataset, target_labels: list[str]) -> "SVNModel":
        """
        Fit SVM models for classification and regression targets.
        If use_multioutput is True, uses MultiOutputClassifier/MultiOutputRegressor for multiple targets.
        Otherwise, trains separate models for each target.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing both input features and target variables
        target_labels : list[str]
            Names of the target variables to predict

        Returns
        -------
        self : returns an instance of self
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
            if self.use_multioutput and len(regression_targets) > 1:
                # Multiple regression targets with multioutput
                y_reg = y[:, regression_targets]
                base_model = SVR()
                # Adapt param grid for multioutput
                multi_param_grid = {
                    f"estimator__{k}": v for k, v in self.regression_param_grid.items()
                }
                multi_model = MultiOutputRegressor(base_model)

                # Use custom multi-target scorer if no scoring is provided
                if self.scoring is None:
                    multi_target_scorer = make_scorer(_multi_target_r2_score)
                else:
                    multi_target_scorer = self.scoring

                grid_search = GridSearchCV(
                    multi_model,
                    multi_param_grid,
                    cv=self.cv_folds,
                    scoring=multi_target_scorer,
                    n_jobs=-1,
                )
                grid_search.fit(X, y_reg)

                self.cv_results_["multi_regression"] = grid_search.cv_results_
                self.best_scores_["multi_regression"] = grid_search.best_score_
                self.best_params_["multi_regression"] = {
                    k.replace("estimator__", ""): v
                    for k, v in grid_search.best_params_.items()
                }

                # Fit final multi-regression model with best parameters
                self.models_["multi_regression"] = grid_search.best_estimator_
            else:
                # Single regression target or separate models
                regression_cols = [
                    col
                    for col, type_ in self.target_types_.items()
                    if type_ == "regression"
                ]
                for i, col in enumerate(regression_cols):
                    y_reg = y[:, regression_targets[i]]
                    base_model = SVR()
                    grid_search = GridSearchCV(
                        base_model,
                        self.regression_param_grid,
                        cv=self.cv_folds,
                        scoring=self.scoring or "r2",
                        n_jobs=-1,
                    )
                    grid_search.fit(X, y_reg)

                    self.cv_results_[f"regression_{col}"] = grid_search.cv_results_
                    self.best_scores_[f"regression_{col}"] = grid_search.best_score_
                    self.best_params_[f"regression_{col}"] = grid_search.best_params_

                    # Fit final regression model
                    self.models_[f"regression_{col}"] = grid_search.best_estimator_

        # Handle classification targets
        if classification_targets:
            if self.use_multioutput and len(classification_targets) > 1:
                # Multiple classification targets with multioutput
                y_clf = y[:, classification_targets]
                base_model = SVC(probability=True)
                # Adapt param grid for multioutput
                multi_param_grid = {
                    f"estimator__{k}": v
                    for k, v in self.classification_param_grid.items()
                }
                multi_model = MultiOutputClassifier(base_model)

                # Use custom multi-target scorer if no scoring is provided
                if self.scoring is None:
                    multi_target_scorer = make_scorer(_multi_target_accuracy_score)
                else:
                    multi_target_scorer = self.scoring

                grid_search = GridSearchCV(
                    multi_model,
                    multi_param_grid,
                    cv=self.cv_folds,
                    scoring=multi_target_scorer,
                    n_jobs=-1,
                )
                grid_search.fit(X, y_clf)

                self.cv_results_["multi_classification"] = grid_search.cv_results_
                self.best_scores_["multi_classification"] = grid_search.best_score_
                self.best_params_["multi_classification"] = {
                    k.replace("estimator__", ""): v
                    for k, v in grid_search.best_params_.items()
                }

                # Fit final multi-classification model with best parameters
                self.models_["multi_classification"] = grid_search.best_estimator_
            else:
                # Single classification target or separate models
                classification_cols = [
                    col
                    for col, type_ in self.target_types_.items()
                    if type_ == "classification"
                ]
                for i, col in enumerate(classification_cols):
                    y_clf = y[:, classification_targets[i]]
                    base_model = SVC(probability=True)
                    grid_search = GridSearchCV(
                        base_model,
                        self.classification_param_grid,
                        cv=self.cv_folds,
                        scoring=self.scoring or "accuracy",
                        n_jobs=-1,
                    )
                    grid_search.fit(X, y_clf.ravel())

                    self.cv_results_[f"classification_{col}"] = grid_search.cv_results_
                    self.best_scores_[f"classification_{col}"] = grid_search.best_score_
                    self.best_params_[f"classification_{col}"] = (
                        grid_search.best_params_
                    )

                    # Fit final classification model
                    self.models_[f"classification_{col}"] = grid_search.best_estimator_

        return self

    def predict(self, X: NDArray | Dataset) -> dict[str, NDArray]:
        """
        Predict using the SVM models.

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
            If models have not been fitted
        """
        if not self.models_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if isinstance(X, Dataset):
            X_processed = self.prepare_features(X)
        else:
            X_processed = X

        predictions = {}

        # Handle predictions based on model type
        for model_name, model in self.models_.items():
            if model_name == "multi_regression":
                # Handle multi-regression predictions
                multi_reg_preds = np.asarray(model.predict(X_processed))
                regression_targets = [
                    col
                    for col, type_ in self.target_types_.items()
                    if type_ == "regression"
                ]
                for i, target in enumerate(regression_targets):
                    predictions[target] = multi_reg_preds[:, i]
            elif model_name == "multi_classification":
                # Handle multi-classification predictions
                multi_clf_preds = np.asarray(model.predict(X_processed))
                classification_targets = [
                    col
                    for col, type_ in self.target_types_.items()
                    if type_ == "classification"
                ]
                for i, target in enumerate(classification_targets):
                    predictions[target] = np.array(
                        [
                            self.target_categories_[target][int(pred)]
                            for pred in multi_clf_preds[:, i]
                        ]
                    )
            elif model_name.startswith("regression_"):
                # Handle single regression predictions
                target = model_name.replace("regression_", "")
                predictions[target] = model.predict(X_processed)
            elif model_name.startswith("classification_"):
                # Handle single classification predictions
                target = model_name.replace("classification_", "")
                preds = model.predict(X_processed)
                predictions[target] = np.array(
                    [self.target_categories_[target][int(pred)] for pred in preds]
                )

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
            If models have not been fitted or if there are no classification targets
        """
        if not any(type_ == "classification" for type_ in self.target_types_.values()):
            raise ValueError("No classification targets available")

        if isinstance(X, Dataset):
            X_processed = self.prepare_features(X)
        else:
            X_processed = X

        probabilities = {}

        # Handle probabilities based on model type
        for model_name, model in self.models_.items():
            if model_name == "multi_classification":
                # Handle multi-classification probabilities
                multi_clf_probs = cast(MultiOutputClassifier, model).predict_proba(
                    X_processed
                )
                classification_targets = [
                    col
                    for col, type_ in self.target_types_.items()
                    if type_ == "classification"
                ]
                if isinstance(multi_clf_probs, list):
                    for i, target in enumerate(classification_targets):
                        probabilities[target] = multi_clf_probs[i]
                else:
                    probabilities[classification_targets[0]] = multi_clf_probs
            elif model_name.startswith("classification_"):
                # Handle single classification probabilities
                target = model_name.replace("classification_", "")
                probabilities[target] = cast(SVC, model).predict_proba(X_processed)

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
            Type of model to plot results for ('regression', 'classification', 'multi_regression', or 'multi_classification')
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
        plt.title(f"SVM {model_type.title()} - {param_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

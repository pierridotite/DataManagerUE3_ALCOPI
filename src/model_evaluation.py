"""
Module for model evaluation and visualization tools.
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, confusion_matrix, classification_report

TargetType = Literal["binary", "continuous", "categorical"]


class ModelEvaluator:
    """Class for evaluating and visualizing model predictions against true values."""

    def __init__(
        self,
        y_true: Union[pd.DataFrame, np.ndarray],
        y_pred: Union[pd.DataFrame, np.ndarray],
        target_names: Optional[List[str]] = None,
        show_plots: bool = True,
    ) -> None:
        """Initialize the ModelEvaluator with true and predicted values.

        Args:
            y_true: True values, can be DataFrame or numpy array
            y_pred: Predicted values, must match y_true dimensions
            target_names: Optional list of target names. If not provided and y_true is a DataFrame,
                        column names will be used.
            show_plots: Whether to display plots interactively. Set to False for testing.

        Raises:
            ValueError: If dimensions don't match or data types are incompatible
            TypeError: If input types are not supported
        """
        self.show_plots = show_plots
        # Handle target names
        if isinstance(y_true, pd.DataFrame):
            self.target_names = target_names or y_true.columns.tolist()
        elif isinstance(y_true, np.ndarray):
            if target_names is None:
                self.target_names = [f"target_{i}" for i in range(y_true.shape[1])]
            else:
                self.target_names = target_names
        else:
            raise TypeError("y_true must be either pandas DataFrame or numpy array")

        # Convert numpy arrays to DataFrame if necessary
        if isinstance(y_true, np.ndarray):
            y_true_df = pd.DataFrame(y_true, columns=self.target_names)
        else:
            y_true_df = y_true
        if isinstance(y_pred, np.ndarray):
            y_pred_df = pd.DataFrame(y_pred, columns=self.target_names)
        elif isinstance(y_pred, pd.DataFrame):
            y_pred_df = y_pred
        else:
            raise TypeError("y_pred must be either pandas DataFrame or numpy array")

        # Validate dimensions
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
            )

        # Validate number of target names matches number of columns
        if len(self.target_names) != y_true.shape[1]:
            raise ValueError(
                f"Number of target names ({len(self.target_names)}) "
                f"does not match number of columns ({y_true.shape[1]})"
            )

        # Initialize dictionaries to store target types and indices
        self.binary_targets: Dict[str, int] = {}
        self.continuous_targets: Dict[str, int] = {}
        self.categorical_targets: Dict[str, int] = {}

        # Create a new DataFrame to store the data with appropriate types
        data_dict = {}

        for idx, col in enumerate(self.target_names):
            dtype = y_true_df[col].dtype
            if pd.api.types.is_bool_dtype(dtype) or (
                isinstance(y_true_df[col].dtype, pd.CategoricalDtype)
                and len(y_true_df[col].unique()) <= 2
            ):
                self.binary_targets[col] = idx
            elif pd.api.types.is_numeric_dtype(dtype):
                self.continuous_targets[col] = idx
            else:
                self.categorical_targets[col] = idx
        # Store data as DataFrames with appropriate types
        self.y_true = y_true_df
        self.y_pred = y_pred_df

    def display_r2_summary(self) -> None:
        """Display R² scores for all targets."""
        if not self.continuous_targets:
            return
        print("\nR² Scores Summary:")
        print("-" * 40)
        for target in self.target_names:
            if target in self.continuous_targets:
                r2 = r2_score(
                    self.y_true[target].to_numpy(), self.y_pred[target].to_numpy()
                )
                print(f"{target:20s}: {r2:.3f}")

    def plot_confusion_matrices(self, figsize: tuple[int, int] = (8, 6)) -> None:
        """Plot confusion matrices for binary and categorical targets.

        Args:
            figsize: Figure size for each confusion matrix plot
        """
        if not (self.binary_targets or self.categorical_targets):
            print("No binary or categorical targets found in the dataset.")
            return

        # Handle binary targets
        for target, idx in self.binary_targets.items():
            # Round predictions to get binary classification
            y_pred_binary = np.round(self.y_pred[target].to_numpy())
            conf_matrix = confusion_matrix(
                self.y_true[target].to_numpy().astype(int), y_pred_binary.astype(int)
            )

            # Plot confusion matrix
            plt.figure(figsize=figsize)
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non-" + target, target],
                yticklabels=["Non-" + target, target],
            )
            plt.title(f"Confusion Matrix for {target}")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            # Print classification metrics
            true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()
            total = np.sum(conf_matrix)

            print(f"\nClassification metrics for {target}:")
            print("-" * 40)
            print(f"Accuracy: {(true_pos + true_neg) / total:.3f}")
            print(
                f"Sensitivity (True Positive Rate): {true_pos / (true_pos + false_neg):.3f}"
            )
            print(
                f"Specificity (True Negative Rate): {true_neg / (true_neg + false_pos):.3f}"
            )
            if self.show_plots:
                plt.show()

        # Handle categorical targets
        for target, idx in self.categorical_targets.items():
            # Get unique classes from true values
            true_classes = set(self.y_true[target].astype(str).unique())
            # Convert predictions to string and get unique values
            pred_classes = set(self.y_pred[target].astype(str).unique())
            # Combine and sort classes
            classes = sorted(true_classes | pred_classes)

            # Convert both arrays to string type for comparison
            y_true_str = self.y_true[target].astype(str).to_numpy()
            y_pred_str = self.y_pred[target].astype(str).to_numpy()

            # Compute confusion matrix
            conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=classes)

            # Plot confusion matrix
            plt.figure(figsize=figsize)
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
            )
            plt.title(f"Confusion Matrix for {target}")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            # Print classification metrics
            total = np.sum(conf_matrix)
            correct = np.sum(np.diag(conf_matrix))

            print(f"\nClassification metrics for {target}:")
            print("-" * 40)
            print(classification_report(y_true_str, y_pred_str, target_names=classes))

            if self.show_plots:
                plt.show()

    def plot_continuous_targets(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Analyze continuous targets with detailed metrics and plots.

        Args:
            figsize: Figure size for each scatter plot
        """
        if not self.continuous_targets:
            print("No continuous targets found in the dataset.")
            return

        for target, idx in self.continuous_targets.items():
            # Get numpy arrays for calculations
            y_true_values = self.y_true[target].to_numpy()
            y_pred_values = self.y_pred[target].to_numpy()

            # Calculate regression metrics
            r2 = r2_score(y_true_values, y_pred_values)
            mse = np.mean((y_true_values - y_pred_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true_values - y_pred_values))

            print(f"\nRegression metrics for {target}:")
            print("-" * 40)
            print(f"R²: {r2:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"MAE: {mae:.3f}")

            # Plot predicted vs actual values
            plt.figure(figsize=figsize)
            plt.scatter(y_true_values, y_pred_values, alpha=0.5)

            # Add perfect prediction line
            min_val = min(np.min(y_true_values), np.min(y_pred_values))
            max_val = max(np.max(y_true_values), np.max(y_pred_values))
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                label="Perfect prediction",
            )

            plt.xlabel(f"Actual {target}")
            plt.ylabel(f"Predicted {target}")
            plt.title(f"Predicted vs Actual Values for {target}\nR² = {r2:.3f}")
            plt.legend()
            plt.grid(True)
            if self.show_plots:
                plt.show()

    def evaluate_all(self) -> None:
        """Run all evaluation methods in sequence."""
        self.display_r2_summary()
        self.plot_confusion_matrices()
        self.plot_continuous_targets()

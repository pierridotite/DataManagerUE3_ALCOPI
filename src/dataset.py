"""
Dataset class for handling hyperspectral data with flexible input types.
"""

from typing import Any, cast, overload
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from numpy.typing import DTypeLike, NDArray
import distinctipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import prince
import seaborn as sns

from .utils import (
    convert_to_str_labels,
    convert_labels_to_indices,
    convert_indices_to_labels,
    is_castable_to_float,
)


class InputType(Enum):
    """Types of input variables."""

    FUNCTIONAL = "functional"  # Variables with order and local correlation
    NON_FUNCTIONAL = "non_functional"  # Variables without specific order/correlation (categorical or numerical)


class GroupConfig:
    """Configuration for a group of variables.

    Attributes:
        features_label: List of variable names in the group
        type: Type of variables (functional or non-functional), defaults to functional
        group_name: Optional name for the group
    """

    def __init__(
        self,
        features_label: list[str] | None = None,
        type: InputType = InputType.FUNCTIONAL,
        group_name: str | None = None,
    ):
        self.features_label = features_label
        self.type = type
        self.group_name = group_name

    def __repr__(self) -> str:
        return (
            f"GroupConfig(features={self.features_label}, "
            f"type={self.type.value}, "
            f"name={self.group_name})"
        )


class Dataset:
    """A class to handle hyperspectral datasets with flexible input types.

    This class can handle data input either as a path to a CSV file or as a pandas DataFrame,
    and can separate variables into input (explanatory) and output (target) variables.
    Input variables can be grouped as either functional (with order and local correlation)
    or non-functional (categorical or numerical without specific order).

    Attributes:
        data (pd.DataFrame): The spectral data
        input_labels (list[str]): The input/explanatory variables labels
        output_labels (list[str]): The output/target variables labels
        n_samples (int): Number of samples in the dataset
        n_features (int): Number of features (wavelengths) in the dataset
        input_groups (list[GroupConfig]): Groups of input variables
    """

    def __init__(
        self,
        data: pd.DataFrame | str | tuple[str, dict[str, Any]],
        input_labels: list[str] | list[int] | None = None,
        output_labels: list[str] | list[int] | None = None,
        input_groups: list[GroupConfig] | None = None,
    ) -> None:
        """Initialize Dataset with data and optional grouping configuration.

        Args:
            data: DataFrame containing all data, or path to CSV file, or tuple of (path, read_csv params)
            input_labels: Column names or indices for input variables
            output_labels: Column names or indices for output variables
            input_groups: Optional list of GroupConfig objects defining variable groups

        Raises:
            TypeError: If data is not a DataFrame, string path, or tuple of (path, params)
            ValueError: If any group has no features
            ValueError: If any group has no name
            ValueError: If input variables are not assigned to exactly one group
            ValueError: If functional variables are not numeric
        """
        # Load data if necessary
        if isinstance(data, (str, tuple)):
            if isinstance(data, str):
                self.data = pd.read_csv(data)
            else:
                path, params = data
                self.data = pd.read_csv(path, **params)
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "data must be a pandas DataFrame, a path to a CSV file, or a tuple of (path, params)"
                )
            self.data = data.copy()

        # Convert input and output labels to string format
        self.input_labels = convert_to_str_labels(self.data, input_labels)
        self.output_labels = convert_to_str_labels(self.data, output_labels)

        # If no input_labels provided, use all columns except output_labels
        if input_labels is None and output_labels is not None:
            self.input_labels = [
                col for col in self.data.columns if col not in self.output_labels
            ]

        # If no output_labels provided, use all columns except input_labels
        if output_labels is None and input_labels is not None:
            self.output_labels = [
                col for col in self.data.columns if col not in self.input_labels
            ]

        # Process input groups
        if input_groups is None:
            # Default: all inputs in one functional group
            self.input_groups = [
                GroupConfig(
                    features_label=self.input_labels,
                    type=InputType.FUNCTIONAL,
                    group_name="all_inputs",
                )
            ]
        else:
            # Validate groups
            for group in input_groups:
                # Verify group has a name
                if group.group_name is None:
                    raise ValueError("All groups must have a name")

                # Verify group has features
                if group.features_label is None or not group.features_label:
                    raise ValueError(f"Group {group.group_name} has no features")

            # Validate that all input variables are assigned to exactly one group
            assigned_vars = set()
            for group in input_groups:
                if group.features_label is not None:
                    assigned_vars.update(group.features_label)

            if assigned_vars != set(self.input_labels):
                unassigned = set(self.input_labels) - assigned_vars
                over_assigned = assigned_vars - set(self.input_labels)
                if unassigned:
                    raise ValueError(
                        f"Input variables not assigned to any group: {unassigned}"
                    )
                if over_assigned:
                    raise ValueError(
                        f"Variables in groups but not in input_labels: {over_assigned}"
                    )

            self.input_groups = input_groups

        # Verify numeric type only for functional variables
        functional_vars: list[str] = []
        for group in self.input_groups:
            if group.type == InputType.FUNCTIONAL and group.features_label is not None:
                functional_vars.extend(group.features_label)

        # Check if functional variables are castable to float
        for var in functional_vars:
            if not all(is_castable_to_float(val) for val in self.data[var]):
                raise ValueError(
                    f"Functional variable '{var}' must be castable to float"
                )

        # Set basic attributes
        self.n_samples = len(self.data)
        self.n_features = self.data.shape[1]

    def get_labels_indices(self, labels: list[str]) -> list[int]:
        """Get the indices of the given labels in the DataFrame.

        Args:
            labels: List of column names

        Returns:
            list[int]: List of column indices
        """
        return convert_labels_to_indices(self.data, labels)

    def get_indices_labels(self, indices: list[int]) -> list[str]:
        """Get the labels corresponding to the given indices in the DataFrame.

        Args:
            indices: List of column indices

        Returns:
            list[str]: List of column names
        """
        return convert_indices_to_labels(self.data, indices)

    def get_data(self) -> pd.DataFrame:
        """Get the spectral data.

        Returns:
            pd.DataFrame: The spectral data
        """
        return self.data

    @overload
    def get_input_labels(self, as_indices: bool = False) -> list[str]: ...

    @overload
    def get_input_labels(self, as_indices: bool = True) -> list[int]: ...

    def get_input_labels(self, as_indices: bool = False) -> list[str] | list[int]:
        """Get the input labels.

        Args:
            as_indices: If True, return indices instead of names

        Returns:
            list[str] | list[int]: The input labels as either names or indices
        """
        if as_indices:
            return convert_labels_to_indices(self.data, self.input_labels)
        return self.input_labels

    @overload
    def get_output_labels(self, as_indices: bool = False) -> list[str]: ...

    @overload
    def get_output_labels(self, as_indices: bool = True) -> list[int]: ...

    def get_output_labels(self, as_indices: bool = False) -> list[str] | list[int]:
        """Get the output labels.

        Args:
            as_indices: If True, return indices instead of names

        Returns:
            list[str] | list[int]: The output labels as either names or indices
        """
        if as_indices:
            return convert_labels_to_indices(self.data, self.output_labels)
        return self.output_labels

    def get_input_data(self) -> pd.DataFrame:
        """Get the input data (features/wavelengths).

        Returns:
            pd.DataFrame: DataFrame of input data
        """
        return self.data[self.input_labels]

    def get_output_data(self, target_labels: list[str] | None = None) -> pd.DataFrame:
        """Get the output data for specified target labels.

        Args:
            target_labels: Specific output labels to return. If None, return all output labels.

        Returns:
            pd.DataFrame: DataFrame of output data

        Raises:
            ValueError: If any target label is not in output_labels
        """
        if target_labels is None:
            return self.data[self.output_labels]

        # Verify target labels are valid
        invalid_labels = set(target_labels) - set(self.output_labels)
        if invalid_labels:
            raise ValueError(f"Invalid target labels: {invalid_labels}")

        return self.data[target_labels]

    def get_sample(self, index: int) -> tuple:
        """Get a single sample and its labels.

        Args:
            index: Index of the sample to retrieve

        Returns:
            tuple: (sample data, input data, output data)

        Raises:
            IndexError: If index is out of bounds
        """
        if 0 <= index < self.n_samples:
            sample = self.data.iloc[index]
            input_data = sample[self.input_labels]
            output_data = sample[self.output_labels]
            return sample, input_data, output_data
        raise IndexError(
            f"Index {index} is out of bounds for dataset of size {self.n_samples}"
        )

    def get_subset(self, indices: list[int]) -> "Dataset":
        """Get a subset of the dataset.

        Args:
            indices: List of indices to include in the subset

        Returns:
            Dataset: A new Dataset instance containing only the specified samples
        """
        subset_data = self.data.iloc[indices]
        return Dataset(subset_data, self.input_labels, self.output_labels)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return self.n_samples

    def __str__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            str: Description of the dataset
        """
        desc = [f"Dataset with {self.n_samples} samples and {self.n_features} features"]
        desc.append(f"Input features: {self.input_labels}")
        desc.append(f"Output features: {self.output_labels}")
        return "\n".join(desc)

    def __repr__(self) -> str:
        """Get a detailed string representation of the dataset.

        Returns:
            str: Detailed description of the dataset
        """
        return (
            f"Dataset(n_samples={self.n_samples}, "
            f"n_features={self.n_features}, "
            f"n_input_features={len(self.input_labels)}, "
            f"n_output_features={len(self.output_labels)})"
        )

    def _create_subplot_figure(
        self, n_vars: int, common_subplot_params: dict[str, Any] | None
    ) -> tuple[Figure, np.ndarray]:
        """Create a figure with subplots based on the number of variables.

        Args:
            n_vars: Number of variables to plot
            common_subplot_params: Optional common parameters for subplots

        Returns:
            tuple containing:
                - matplotlib Figure
                - numpy array of Axes
        """
        # Calculate default subplot parameters
        default_ncols = min(2, n_vars)
        if common_subplot_params and "ncols" in common_subplot_params:
            ncols = min(common_subplot_params["ncols"], n_vars)
        else:
            ncols = default_ncols

        if common_subplot_params and "nrows" in common_subplot_params:
            nrows = common_subplot_params["nrows"]
        else:
            nrows = (n_vars - 1) // ncols + 1

        # Calculate figure size
        width_per_plot = 6
        height_per_plot = 4
        total_width = width_per_plot * ncols
        total_height = height_per_plot * nrows

        # Setup subplot parameters
        subplot_params = {
            "nrows": nrows,
            "ncols": ncols,
            "figsize": (total_width, total_height),
        }

        # Update with common parameters if provided
        if common_subplot_params:
            subplot_params.update(common_subplot_params)
            subplot_params["ncols"] = min(subplot_params["ncols"], n_vars)

        # Create figure and axes
        fig, axes = plt.subplots(**subplot_params)

        # Ensure axes is always a flattened array
        if n_vars == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        return fig, axes

    def _plot_spectra(
        self, ax: Axes, var: str, x_labels: list[str], is_qualitative: bool
    ) -> None:
        """Plot spectra on given axes, colored by variable.

        Args:
            ax: Matplotlib axes to plot on
            var: Variable name to color by
            x_labels: Labels for x-axis (typically wavelengths)
            is_qualitative: Whether the variable is qualitative
        """
        # Create array for x-axis values
        try:
            x_values = [float(x) for x in x_labels]
        except ValueError:
            x_values = range(len(x_labels))

        output_values = self.data[var]

        if is_qualitative:
            # Qualitative coloring
            unique_values = np.sort(output_values.unique())
            colors = distinctipy.get_colors(
                len(unique_values), pastel_factor=0.7, rng=0
            )
            color_dict = dict(zip(unique_values, colors))

            for idx in range(len(self.data)):
                spectrum = self.data[x_labels].iloc[idx]
                value = output_values.iloc[idx]
                ax.plot(
                    x_values,
                    spectrum,
                    color=color_dict[value],
                    alpha=0.3,
                    label=f"Class {value}",
                )

            # Add legend for qualitative plots
            handles, labels = ax.get_legend_handles_labels()

            # Trier les labels et handles en même temps
            sorted_pairs = sorted(zip(labels, handles), key=lambda pair: pair[0])
            by_label = dict(sorted_pairs)
            ax.legend(by_label.values(), by_label.keys())
        else:
            # Quantitative coloring
            custom_cmap = plt.get_cmap("plasma")
            norm = Normalize(output_values.min(), output_values.max())

            for idx in range(len(self.data)):
                spectrum = self.data[x_labels].iloc[idx]
                color = custom_cmap(norm(output_values.iloc[idx]))
                ax.plot(x_values, spectrum, color=color, alpha=0.5)

            # Add colorbar for quantitative plots
            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=var)

        # Common axis settings
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Spectral Data colored by {var}")
        ax.grid(True, alpha=0.3)

    def _plot_functional_data(
        self,
        x_labels: list[str],
        selected_labels: list[str],
        common_subplot_params: dict[str, Any] | None,
        group_labels: str | list[str] | None = None,
    ) -> None:
        """Plot functional data using spectral visualization.

        Args:
            x_labels: Labels for x-axis (typically wavelengths)
            selected_labels: Labels to use for coloring the plots
            common_subplot_params: Common parameters for all subplots
            group_labels: Optional group name(s) to filter by. If None, all functional groups are plotted.
                        Can be a single group name or a list of group names.

        Raises:
            ValueError: If any specified group name is not found
        """
        # Validate and normalize group names
        if group_labels is not None:
            if isinstance(group_labels, str):
                group_labels = [group_labels]

            # Verify all group names exist
            all_groups = set(self.get_group_names())
            invalid_groups = set(group_labels) - all_groups
            if invalid_groups:
                raise ValueError(f"Invalid group names: {invalid_groups}")

        # Separate qualitative and quantitative variables
        qual_vars = []
        quant_vars = []
        for label in selected_labels:
            if pd.api.types.is_numeric_dtype(self.data[label]):
                quant_vars.append(label)
            else:
                qual_vars.append(label)

        # Sort variables for consistent display order
        qual_vars.sort()
        quant_vars.sort()

        # Get functional data groups
        functional_groups = self.get_type_data(InputType.FUNCTIONAL)

        # For each functional group, create plots
        for group in functional_groups:
            # Skip if group is not in the filtered list
            if group_labels is not None and group.group_name not in group_labels:
                continue

            # Plot qualitative variables
            if qual_vars:
                fig_qual, axes_qual = self._create_subplot_figure(
                    len(qual_vars), common_subplot_params
                )

                for ax, var in zip(axes_qual, qual_vars):
                    self._plot_spectra(ax, var, x_labels, is_qualitative=True)
                    ax.set_title(f"{group.group_name} - Colored by {var}")

                for ax in axes_qual[len(qual_vars) :]:
                    ax.set_visible(False)

                plt.tight_layout()

            # Plot quantitative variables
            if quant_vars:
                fig_quant, axes_quant = self._create_subplot_figure(
                    len(quant_vars), common_subplot_params
                )

                for ax, var in zip(axes_quant, quant_vars):
                    self._plot_spectra(ax, var, x_labels, is_qualitative=False)
                    ax.set_title(f"{group.group_name} - Colored by {var}")

                for ax in axes_quant[len(quant_vars) :]:
                    ax.set_visible(False)

                plt.tight_layout()

    def _plot_non_functional_data(
        self,
        non_functional_groups: list[GroupConfig],
        selected_labels: list[str],
    ) -> None:
        """Plot non-functional data using dimensionality reduction visualization.
        The method automatically chooses between:
        - PCA for all numerical variables
        - MCA for all categorical variables
        - FAMD for mixed variables

        Args:
            non_functional_groups: List of non-functional GroupConfig objects
            selected_labels: Labels to color the plots by
        """
        # Combine all non-functional data into a DataFrame
        combined_data = pd.DataFrame()
        for group in non_functional_groups:
            group_data = self.data[group.features_label]
            combined_data = pd.concat([combined_data, group_data], axis=1)

        # Check variable types
        is_numeric = [
            pd.api.types.is_numeric_dtype(combined_data[col])
            for col in combined_data.columns
        ]
        all_numeric = all(is_numeric)
        all_categorical = all(not x for x in is_numeric)

        # Determine number of components (minimum between 2 and number of features)
        n_components = min(2, combined_data.shape[1])
        if n_components < 1:
            raise ValueError(
                "At least one feature is required for dimensionality reduction"
            )

        # Choose appropriate method
        if all_numeric:
            # Use PCA for numerical variables
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(combined_data)
            pca = PCA(n_components=n_components, random_state=42)
            result = pca.fit_transform(scaled_data)
            explained_variance_ratio = pca.explained_variance_ratio_
            method_name = "PCA"
            result_df: pd.DataFrame = pd.DataFrame(
                result,
                columns=[f"Component {i+1}" for i in range(n_components)],
                index=combined_data.index,
            )
        elif all_categorical:
            # Use MCA for categorical variables
            mca = prince.MCA(n_components=n_components, random_state=42)
            result_df = pd.DataFrame(mca.fit_transform(combined_data))
            # Convert percentage strings to float numbers between 0 and 1
            explained_variance_ratio = (
                mca.eigenvalues_summary["% of variance"].str.rstrip("%").astype(float)
                / 100
            )
            method_name = "MCA"
        else:
            # Use FAMD for mixed variables
            famd = prince.FAMD(n_components=n_components, random_state=42)
            result_df = pd.DataFrame(famd.fit_transform(combined_data))
            explained_variance_ratio = (
                famd.eigenvalues_summary["% of variance"].str.rstrip("%").astype(float)
                / 100
            )
            method_name = "FAMD"

        # Calculate cumulative variance ratio
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Plot results for each label
        for label in selected_labels:
            plt.figure(figsize=(10, 8))

            if pd.api.types.is_numeric_dtype(self.data[label]):
                # Quantitative coloring
                if n_components == 1:
                    # Pour une seule composante, créer un scatter plot avec la composante en x et une valeur constante en y
                    scatter = plt.scatter(
                        result_df.iloc[:, 0],
                        np.zeros_like(result_df.iloc[:, 0]),
                        c=self.data[label],
                        cmap="viridis",
                        alpha=0.6,
                    )
                else:
                    scatter = plt.scatter(
                        result_df.iloc[:, 0],
                        result_df.iloc[:, 1],
                        c=self.data[label],
                        cmap="viridis",
                        alpha=0.6,
                    )
                plt.colorbar(scatter, label=label)
            else:
                # Qualitative coloring
                unique_values = np.sort(self.data[label].unique())
                colors = distinctipy.get_colors(
                    len(unique_values), pastel_factor=0.7, rng=0
                )
                color_dict = dict(zip(unique_values, colors))

                # Keep track of legend handles and labels
                legend_elements = []

                for value in unique_values:
                    mask = self.data[label] == value
                    filtered_data = result_df[mask]
                    if n_components == 1:
                        # Pour une seule composante, créer un scatter plot avec la composante en x et une valeur constante en y
                        scatter = plt.scatter(
                            filtered_data[filtered_data.columns[0]],
                            np.zeros_like(filtered_data[filtered_data.columns[0]]),
                            c=[color_dict[value]],
                            label=f"{value}",
                            alpha=0.6,
                        )
                    else:
                        scatter = plt.scatter(
                            filtered_data[filtered_data.columns[0]],
                            filtered_data[filtered_data.columns[1]],
                            c=[color_dict[value]],
                            label=f"{value}",
                            alpha=0.6,
                        )
                    legend_elements.append(scatter)

                # Only add legend if there are elements to show
                if legend_elements:
                    plt.legend()

            # Add explained variance information
            variance_text = (
                f"Explained {'variance' if method_name == 'PCA' else 'inertia'} ratio:\n"
                f"Dim 1: {explained_variance_ratio[0]:.3f}\n"
            )
            if n_components > 1:
                variance_text += (
                    f"Dim 2: {explained_variance_ratio[1]:.3f}\n"
                    f"Cumulative: {cumulative_variance_ratio[1]:.3f}"
                )
            plt.text(
                0.02,
                0.98,
                variance_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Set labels and title
            dim_type = "Component" if method_name == "PCA" else "Dimension"
            plt.xlabel(
                f"{dim_type} 1 ({explained_variance_ratio[0]:.1%} explained {'variance' if method_name == 'PCA' else 'inertia'})"
            )
            if n_components > 1:
                plt.ylabel(
                    f"{dim_type} 2 ({explained_variance_ratio[1]:.1%} explained {'variance' if method_name == 'PCA' else 'inertia'})"
                )
            else:
                plt.ylabel("No second component (single feature)")
            plt.title(f"{method_name} of Non-Functional Variables Colored by {label}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    def show(
        self,
        output_labels: str | list[str] | None = None,
        group_labels: str | list[str] | None = None,
        common_subplot_params: dict[str, Any] | None = None,
    ) -> None:
        """Display the dataset with different visualizations based on input types.

        Args:
            output_labels: Labels to color the data by
            group_labels: Group names to display. If None, all groups are displayed.
                         If a string, only that group is displayed.
                         If a list of strings, only those groups are displayed.
            common_subplot_params: Common parameters for all subplots

        Raises:
            ValueError: If output_labels contains invalid labels
            ValueError: If group_labels contains invalid group names
            ValueError: If a group has no features
            ValueError: If subplot parameters are invalid
        """
        # Validate output labels
        if output_labels is not None:
            if isinstance(output_labels, str):
                if output_labels not in self.output_labels:
                    raise ValueError(f"Invalid output label: {output_labels}")
                selected_labels = [output_labels]
            else:
                invalid_labels = [
                    label for label in output_labels if label not in self.output_labels
                ]
                if invalid_labels:
                    raise ValueError(f"Invalid output labels: {invalid_labels}")
                selected_labels = output_labels
        else:
            selected_labels = self.output_labels

        # Validate subplot parameters
        if common_subplot_params is not None:
            if "ncols" in common_subplot_params and common_subplot_params["ncols"] <= 0:
                raise ValueError("Number of columns must be positive")
            if "nrows" in common_subplot_params and common_subplot_params["nrows"] <= 0:
                raise ValueError("Number of rows must be positive")

        # Recursive approach for group_labels handling
        if group_labels is None:
            # If no groups specified, use all groups
            return self.show(
                output_labels=output_labels,
                common_subplot_params=common_subplot_params,
                group_labels=self.get_group_names(),
            )

        if isinstance(group_labels, list):
            # Verify all group names exist
            all_groups = set(self.get_group_names())
            invalid_groups = set(group_labels) - all_groups
            if invalid_groups:
                raise ValueError(f"Invalid group names: {invalid_groups}")

            # Process each group individually
            for group_name in group_labels:
                self.show(
                    output_labels=output_labels,
                    common_subplot_params=common_subplot_params,
                    group_labels=group_name,
                )
            return

        # At this point, group_labels is a single string
        group_name = group_labels

        # Verify group exists
        if group_name not in self.get_group_names():
            raise ValueError(f"Group {group_name} not found")

        # Find the group configuration
        group = next(g for g in self.input_groups if g.group_name == group_name)

        # Verify group has features
        if group.features_label is None or not group.features_label:
            raise ValueError(f"Group {group_name} has no features")

        # Plot based on group type
        if group.type == InputType.FUNCTIONAL:
            self._plot_functional_data(
                group.features_label,
                selected_labels,
                common_subplot_params,
                group_labels,
            )
        else:  # InputType.NON_FUNCTIONAL
            self._plot_non_functional_data([group], selected_labels)

    def get_group_data(self, group_name: str) -> np.ndarray:
        """Get data for a specific group of variables.

        Args:
            group_name: Name of the group

        Returns:
            np.ndarray: Array of data for the specified group

        Raises:
            ValueError: If group_name is not found
        """
        for group in self.input_groups:
            if group.group_name == group_name:
                return np.asarray(self.data[group.features_label].values)
        raise ValueError(f"Group {group_name} not found in input groups")

    def get_type_data(self, var_type: InputType) -> list[GroupConfig]:
        """Get all groups of a specific type.

        Args:
            var_type: Type of variables to retrieve

        Returns:
            list[GroupConfig]: List of GroupConfig objects matching the specified type
        """
        return [group for group in self.input_groups if group.type == var_type]

    def get_group_names(self, var_type: InputType | None = None) -> list[str]:
        """Get names of all groups, optionally filtered by type.

        Args:
            var_type: Optional type to filter by

        Returns:
            list[str]: Names of matching groups

        Raises:
            ValueError: If any group has no name
        """
        if var_type is None:
            names = [
                group.group_name
                for group in self.input_groups
                if group.group_name is not None
            ]
            if len(names) != len(self.input_groups):
                raise ValueError("All groups must have a name")
            return names

        names = [
            group.group_name
            for group in self.input_groups
            if group.type == var_type and group.group_name is not None
        ]
        if len(names) != len([g for g in self.input_groups if g.type == var_type]):
            raise ValueError("All groups must have a name")
        return names

    def add(
        self,
        data: pd.DataFrame | str | tuple[str, dict[str, Any]],
        input_labels: list[str] | list[int] | None = None,
        output_labels: list[str] | list[int] | None = None,
        input_groups: list[GroupConfig] | None = None,
        by: list[str] | None = None,
    ) -> None:
        """Add new data to the dataset.

        Args:
            data: DataFrame containing new data, or path to CSV file, or tuple of (path, read_csv params)
            input_labels: Column names or indices for input variables in new data
            output_labels: Column names or indices for output variables in new data
            input_groups: Optional list of GroupConfig objects defining variable groups for new data
            by: Optional list of columns to merge on

        Raises:
            TypeError: If data is not a DataFrame, string path, or tuple of (path, params)
        """
        # Create temporary dataset with new data
        if isinstance(data, (str, tuple)):
            if isinstance(data, str):
                temp_data = pd.read_csv(data)
            else:
                path, params = data
                temp_data = pd.read_csv(path, **params)
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "data must be a pandas DataFrame, a path to a CSV file, or a tuple of (path, params)"
                )
            temp_data = data.copy()

        temp_dataset = Dataset(
            temp_data,
            input_labels=input_labels or self.input_labels,
            output_labels=output_labels or self.output_labels,
            input_groups=input_groups or self.input_groups,
        )

        # If no merge columns specified, use index
        if by is None:
            self.data = pd.concat([self.data, temp_dataset.data])
        else:
            # Merge on specified columns
            self.data = pd.merge(self.data, temp_dataset.data, on=by, how="outer")

        # Update labels if new ones were provided
        if input_labels is not None:
            self.input_labels = convert_to_str_labels(self.data, input_labels)
        if output_labels is not None:
            self.output_labels = convert_to_str_labels(self.data, output_labels)
        if input_groups is not None:
            self.input_groups = input_groups

        # Update basic attributes
        self.n_samples = len(self.data)
        self.n_features = self.data.shape[1]

    def display_categorical_distribution(
        self,
        variables: str | list[str] | None = None,
    ) -> None:
        """Display a heatmap showing the distribution of categorical variables.

        The heatmap shows the count of occurrences for each category in each categorical variable.
        Categories are shown on the x-axis (grouped by variable) and variables on the y-axis,
        creating a block diagonal structure.
        By default, only shows categorical variables from input_labels and output_labels.

        Args:
            variables: Optional variable name(s) to include in the visualization.
                      If None, all categorical variables from input_labels and output_labels are included.
                      Can be either input or output variables.

        Raises:
            ValueError: If any specified variable is not found in the dataset
            ValueError: If any specified variable is not categorical
            ValueError: If no categorical variables are found
        """
        # Validate and process input variables
        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]

            # Check if all variables exist in the dataset
            invalid_vars = [var for var in variables if var not in self.data.columns]
            if invalid_vars:
                raise ValueError(f"Variables not found in dataset: {invalid_vars}")

            # Filter categorical variables
            cat_vars = [
                var
                for var in variables
                if not pd.api.types.is_numeric_dtype(self.data[var])
            ]

            if not cat_vars:
                raise ValueError(
                    "No categorical variables found among specified variables"
                )

            non_cat_vars = set(variables) - set(cat_vars)
            if non_cat_vars:
                raise ValueError(
                    f"The following variables are not categorical: {non_cat_vars}"
                )
        else:
            # Get all categorical variables from input_labels and output_labels
            all_relevant_vars = set(self.input_labels) | set(self.output_labels)
            cat_vars = [
                col
                for col in all_relevant_vars
                if not pd.api.types.is_numeric_dtype(self.data[col])
            ]
            if not cat_vars:
                raise ValueError(
                    "No categorical variables found in input or output labels"
                )

        # Sort variables for consistent display
        cat_vars = sorted(cat_vars)

        # Create an empty DataFrame to store the counts
        all_counts = pd.DataFrame()

        # Process each variable
        for var in cat_vars:
            # Get value counts for this variable
            counts = self.data[var].value_counts()

            # Create column names by prefixing categories with variable name
            new_columns = {val: f"{var}_{val}" for val in counts.index}

            # Create a DataFrame for this variable with renamed columns
            var_df = pd.DataFrame(
                {new_columns[val]: [count] for val, count in counts.items()},
                index=[var],
            )

            # Concatenate with the main DataFrame
            all_counts = pd.concat([all_counts, var_df], axis=0)

        # Fill NaN values with 0
        all_counts = all_counts.fillna(0)

        # Create the heatmap
        plt.figure(
            figsize=(
                max(10, all_counts.shape[1] * 0.8),
                max(6, all_counts.shape[0] * 0.8),
            )
        )

        # Plot heatmap with custom parameters
        sns.heatmap(
            all_counts,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={"label": "Count"},
        )

        plt.title("Distribution des classes par variable catégorielle")
        plt.xlabel("Classe")
        plt.ylabel("Variable")

        # Rotate x-axis labels and adjust their position
        plt.xticks(rotation=45, ha="right")

        # Add more space at the bottom for the labels
        plt.tight_layout()

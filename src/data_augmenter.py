"""Data augmentation utilities for mixed data types (continuous and categorical)."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable


@dataclass
class AugmentationParams:
    """Parameters for data augmentation techniques.

    Attributes:
        mixup_alpha (float): Alpha parameter for beta distribution in mixup (default: 0.2)
        gaussian_noise_std (float): Standard deviation for Gaussian noise (default: 0.01)
        jitter_factor (float): Maximum factor for spectral jittering (default: 0.02)
        augmentation_probability (float): Probability of applying augmentation (default: 0.5)
        by (str | list[str]): Column(s) defining groups for augmentation
        batch_size (int): Number of samples to generate per group
        exclude_columns (list[str] | None): Columns to exclude from augmentation (e.g., IDs)
    """

    mixup_alpha: float = 0.2
    gaussian_noise_std: float = 0.01
    jitter_factor: float = 0.02
    augmentation_probability: float = 0.5
    by: str | list[str] = "class"
    batch_size: int = 100
    exclude_columns: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive")
        if self.gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if self.jitter_factor < 0:
            raise ValueError("jitter_factor must be non-negative")
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Convert by to list if it's a string
        if isinstance(self.by, str):
            self.by = [self.by]


class DataAugmenter:
    """Class for augmenting mixed data types.

    This class provides methods for data augmentation specifically designed for
    mixed data types (continuous and categorical). It includes techniques such as:
    - Mixup: Linear interpolation between pairs of samples within groups
    - Gaussian noise: Addition of random noise to continuous data
    - Jittering: Random intensity variations for continuous data
    - Categorical sampling: Sampling from empirical distributions within groups
    """

    def __init__(self, params: AugmentationParams | None = None):
        """Initialize the augmenter with given parameters.

        Args:
            params: Configuration parameters for augmentation techniques.
                   If None, default parameters will be used.
        """
        self.params = params or AugmentationParams()
        self._categorical_columns: list[str] = []
        self._continuous_columns: list[str] = []
        self._group_empirical_distributions: dict = {}

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify categorical and continuous columns in the DataFrame.

        Args:
            df: Input DataFrame
        """
        exclude = self.params.exclude_columns or []
        for col in df.columns:
            if col in exclude or col in self.params.by:
                continue
            if pd.api.types.is_numeric_dtype(
                df[col]
            ) and not pd.api.types.is_bool_dtype(df[col]):
                self._continuous_columns.append(col)
            else:
                self._categorical_columns.append(col)

    def _compute_empirical_distributions(self, df: pd.DataFrame) -> None:
        """Compute empirical distributions for categorical variables within groups.

        Args:
            df: Input DataFrame
        """
        for cat_col in self._categorical_columns:
            self._group_empirical_distributions[cat_col] = {}
            for group_name, group_df in df.groupby(self.params.by):
                probs = group_df[cat_col].value_counts(normalize=True)
                if isinstance(group_name, tuple):
                    self._group_empirical_distributions[cat_col][group_name] = probs
                else:
                    self._group_empirical_distributions[cat_col][(group_name,)] = probs

    def _augment_continuous(
        self, data: NDArray[np.float32], group_size: int
    ) -> NDArray[np.float32]:
        """Apply augmentation to continuous data.

        Args:
            data: Input continuous data
            group_size: Number of samples to generate

        Returns:
            Augmented continuous data

        Notes:
            Beta distribution is used for mixing weights instead of uniform distribution
            based on the findings from "mixup: Beyond Empirical Risk Minimization"
            (Zhang et al., 2018, ICLR) - https://arxiv.org/abs/1710.09412

            While uniform distribution might seem simpler, Beta distribution with small
            alpha (e.g., 0.2) creates a U-shaped distribution that:
            1. Concentrates samples near the original training points
            2. Still allows for interpolation across the full range
            3. Provides better regularization and generalization than uniform sampling
            4. Helps prevent manifold intrusion (mixing samples from different classes)
        """
        # Generate indices for mixup within the group
        idx1 = np.random.randint(0, len(data), size=group_size)
        idx2 = np.random.randint(0, len(data), size=group_size)

        # Generate mixing weights
        mixing_weights = (
            np.random.beta(
                self.params.mixup_alpha, self.params.mixup_alpha, size=group_size
            )
            .reshape(-1, 1)
            .astype(np.float32)
        )

        # Apply mixup
        augmented = (
            mixing_weights * data[idx1] + (1 - mixing_weights) * data[idx2]
        ).astype(np.float32)

        # Apply noise and jittering if probability condition is met
        if np.random.random() < self.params.augmentation_probability:
            noise = np.random.normal(
                0, self.params.gaussian_noise_std, size=augmented.shape
            ).astype(np.float32)
            jitter = np.random.uniform(
                1 - self.params.jitter_factor,
                1 + self.params.jitter_factor,
                size=augmented.shape,
            ).astype(np.float32)
            augmented = ((augmented + noise) * jitter).astype(np.float32)

        return augmented

    def _augment_categorical(
        self, group_key: tuple, group_size: int, df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """Generate categorical variables based on empirical distributions.

        Args:
            group_key: Key identifying the group
            group_size: Number of samples to generate
            df: Original DataFrame to get data types from

        Returns:
            Dictionary mapping column names to generated values
        """
        result = {}
        for col in self._categorical_columns:
            probs = self._group_empirical_distributions[col][group_key]
            values = np.random.choice(probs.index, size=group_size, p=probs.values)
            # Create a pandas Series with the same dtype as the original column
            result[col] = pd.Series(values, dtype=df[col].dtype)
        return result

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment the input DataFrame.

        Args:
            df: Input DataFrame containing both features and targets

        Returns:
            Augmented DataFrame with the same structure as input

        Raises:
            ValueError: If required columns are missing or if input data is invalid
        """
        # Validate input
        missing_cols = [col for col in self.params.by if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing grouping columns: {missing_cols}")

        # Initialize column type tracking
        self._identify_column_types(df)
        self._compute_empirical_distributions(df)

        # Prepare storage for augmented data
        augmented_data = []

        # Process each group
        for group_name, group_df in df.groupby(self.params.by):
            if not isinstance(group_name, tuple):
                group_name = (group_name,)

            # Convert continuous data to numpy array
            continuous_data = group_df[self._continuous_columns].values.astype(
                np.float32
            )

            # Generate augmented continuous data
            aug_continuous = self._augment_continuous(
                continuous_data, self.params.batch_size
            )

            # Generate categorical data
            aug_categorical = self._augment_categorical(
                group_name, self.params.batch_size, df
            )

            # Combine data for this group
            group_data = pd.DataFrame(aug_continuous, columns=self._continuous_columns)
            for col, values in aug_categorical.items():
                group_data[col] = values

            # Add group columns
            for col, val in zip(self.params.by, group_name):
                group_data[col] = pd.Series(
                    [val] * self.params.batch_size, dtype=df[col].dtype
                )

            augmented_data.append(group_data)

        # Combine all groups
        result = pd.concat(augmented_data, axis=0, ignore_index=True)

        # Ensure same column order as input
        return result[df.columns]

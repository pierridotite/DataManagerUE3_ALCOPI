"""
Utility functions for data manipulation and conversion.
"""

from typing import cast
import pandas as pd
import numpy as np


def convert_to_str_labels(
    df: pd.DataFrame, labels: list[str] | list[int] | None
) -> list[str]:
    """Convert input/output labels to string format.

    Args:
        df: DataFrame containing the data
        labels: Labels to convert, can be None, string indices or integer indices

    Returns:
        list[str]: List of column names
    """
    if labels is None:
        return list(df.columns)

    result: list[str] = []
    for label in labels:
        if isinstance(label, str):
            if label not in df.columns:
                raise ValueError(f"Column {label} not found in DataFrame")
            result.append(label)
        else:
            if not 0 <= label < len(df.columns):
                raise ValueError(f"Column index {label} out of range")
            result.append(df.columns[label])
    return result


def convert_labels_to_indices(df: pd.DataFrame, labels: list[str]) -> list[int]:
    """Convert column names to their integer indices.

    Args:
        df: DataFrame containing the data
        labels: Column names to convert to indices

    Returns:
        list[int]: List of column indices

    Raises:
        ValueError: If any label is not found in the DataFrame
    """
    result: list[int] = []
    for label in labels:
        try:
            idx = cast(int, df.columns.get_loc(label))  # Cast to ensure type safety
            result.append(idx)
        except KeyError:
            raise ValueError(f"Column {label} not found in DataFrame")
    return result


def convert_indices_to_labels(df: pd.DataFrame, indices: list[int]) -> list[str]:
    """Convert integer indices to column names.

    Args:
        df: DataFrame containing the data
        indices: Column indices to convert to names

    Returns:
        list[str]: List of column names

    Raises:
        ValueError: If any index is out of range
    """
    if any(not 0 <= idx < len(df.columns) for idx in indices):
        raise ValueError("One or more indices are out of range")
    return [df.columns[idx] for idx in indices]


def is_castable_to_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _detect_numerical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Detect numerical features in a pandas DataFrame based on data type.
    A feature is considered numerical only if it has a numeric dtype (int or float).
    String representations of numbers are not considered numerical.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where each column is a feature

    Returns
    -------
    np.ndarray
        Array of indices of numerical features
    """
    numerical_mask = []
    for col in df.columns:
        # Check if column is numeric (int or float) and not boolean
        is_numerical = (
            pd.api.types.is_integer_dtype(df[col])
            or pd.api.types.is_float_dtype(df[col])
        ) and not pd.api.types.is_bool_dtype(df[col])

        numerical_mask.append(is_numerical)

    return np.where(np.array(numerical_mask))[0]


def _detect_numerical_features_df(df: pd.DataFrame) -> np.ndarray:
    """
    Detect numerical features in a pandas DataFrame based on data type.
    A feature is considered numerical only if it has a numeric dtype (int or float).
    String representations of numbers are not considered numerical.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where each column is a feature

    Returns
    -------
    np.ndarray
        Array of indices of numerical features
    """
    numerical_mask = []
    for col in df.columns:
        # Check if column is numeric (int or float) and not boolean
        is_numerical = (
            pd.api.types.is_integer_dtype(df[col])
            or pd.api.types.is_float_dtype(df[col])
        ) and not pd.api.types.is_bool_dtype(df[col])

        numerical_mask.append(is_numerical)

    return np.where(np.array(numerical_mask))[0]

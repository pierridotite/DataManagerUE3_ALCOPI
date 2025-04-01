import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter
from typing import Tuple, Optional


class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformer.

    SNV normalizes each spectrum individually by subtracting its mean and dividing
    by its standard deviation. This helps to correct for scatter effects in spectral data.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
    """

    def __init__(self, copy: bool = True):
        self.copy = copy

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SNVTransformer":
        """
        Fit the transformer. Does nothing as SNV is a stateless transformation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SNV normalization to each spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X

        # Calculate mean and standard deviation for each spectrum (row)
        means = np.mean(X_transformed, axis=1, keepdims=True)
        stds = np.std(X_transformed, axis=1, keepdims=True, ddof=1)

        # Apply SNV transformation without creating a new array
        # Use in-place operations to avoid creating a new array when copy=False
        X_transformed -= means  # In-place subtraction
        X_transformed /= stds  # In-place division

        return X_transformed

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X, y).transform(X)


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay filter transformer.

    Applies a Savitzky-Golay filter to smooth and/or differentiate the data.
    This is particularly useful for spectral data to reduce noise while
    preserving peak shapes.

    Parameters
    ----------
    window_length : int, default=9
        The length of the filter window (must be a positive odd integer).
    polyorder : int, default=2
        The order of the polynomial used to fit the samples (must be less than window_length).
    deriv : int, default=0
        The order of the derivative to compute (0 means only smoothing).
    delta : float, default=1.0
        The spacing of the samples to which the filter is applied.
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
    mode : str, default='interp'
        Determines how the edges of the signal are treated.
        See scipy.signal.savgol_filter for details.
    cval : float, default=0.0
        Value to fill past the edges of the input if mode is 'constant'.
    """

    def __init__(
        self,
        window_length: int = 9,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0,
        copy: bool = True,
        mode: str = "interp",
        cval: float = 0.0,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.copy = copy
        self.mode = mode
        self.cval = cval

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "SavitzkyGolayTransformer":
        """
        Fit the transformer. Does nothing as Savitzky-Golay is a stateless transformation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate parameters
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to each spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X
            result_holder = np.empty_like(X_transformed[0])  # For temporary storage

        # Apply Savitzky-Golay filter to each spectrum (row)
        for i in range(X_transformed.shape[0]):
            if self.copy:
                # When copying, we can directly assign
                X_transformed[i] = savgol_filter(
                    X_transformed[i],
                    window_length=self.window_length,
                    polyorder=self.polyorder,
                    deriv=self.deriv,
                    delta=self.delta,
                    mode=self.mode,
                    cval=self.cval,
                )
            else:
                # When not copying, use a temporary array to hold results, then copy back
                np.copyto(
                    result_holder,
                    savgol_filter(
                        X_transformed[i],
                        window_length=self.window_length,
                        polyorder=self.polyorder,
                        deriv=self.deriv,
                        delta=self.delta,
                        mode=self.mode,
                        cval=self.cval,
                    ),
                )
                # Copy result back to original array
                np.copyto(X_transformed[i], result_holder)

        return X_transformed

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X, y).transform(X)


class DerivativeTransformer(BaseEstimator, TransformerMixin):
    """
    Derivative transformer using Savitzky-Golay method.

    Computes the first or second derivative of spectral data using
    the Savitzky-Golay filter. This is useful for enhancing peak detection
    and reducing baseline effects in spectral data.

    Parameters
    ----------
    order : int, default=1
        Order of the derivative (1 for first derivative, 2 for second derivative).
    window_length : int, default=15
        The length of the filter window (must be a positive odd integer).
    polyorder : int, default=2
        The order of the polynomial used to fit the samples (must be less than window_length).
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
    """

    def __init__(
        self,
        order: int = 1,
        window_length: int = 15,
        polyorder: int = 2,
        copy: bool = True,
    ):
        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder
        self.copy = copy

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "DerivativeTransformer":
        """
        Fit the transformer. Does nothing as derivative transformation is stateless.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate parameters
        if self.order not in [1, 2]:
            raise ValueError(
                "order must be 1 (first derivative) or 2 (second derivative)"
            )
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply derivative transformation to each spectrum.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if self.copy:
            X_transformed = X.copy()
        else:
            X_transformed = X
            result_holder = np.empty_like(X_transformed[0])  # For temporary storage

        # Apply Savitzky-Golay derivative to each spectrum (row)
        for i in range(X_transformed.shape[0]):
            if self.copy:
                # When copying, we can directly assign
                X_transformed[i] = savgol_filter(
                    X_transformed[i],
                    window_length=self.window_length,
                    polyorder=self.polyorder,
                    deriv=self.order,
                    mode="interp",
                )
            else:
                # When not copying, use a temporary array to hold results, then copy back
                np.copyto(
                    result_holder,
                    savgol_filter(
                        X_transformed[i],
                        window_length=self.window_length,
                        polyorder=self.polyorder,
                        deriv=self.order,
                        mode="interp",
                    ),
                )
                # Copy result back to original array
                np.copyto(X_transformed[i], result_holder)

        return X_transformed

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The spectral data to transform.
        y : np.ndarray of shape (n_samples,), default=None
            The target values. Not used, present for API consistency.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X, y).transform(X)

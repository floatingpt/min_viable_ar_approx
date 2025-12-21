"""
Time-Varying Multivariate Autoregressive (MVAR) Classifier

Binary classification for locally stationary multivariate time series based on 
second-order structure using MVAR approximations with sieve basis expansion.

Mathematical Framework:
----------------------
Each observation Z_t ∈ R^p follows a time-varying MVAR(b) model:

    Z_t = Σ_{j=1}^b A_j(t) Z_{t-j} + ε_t

where A_j(t) ∈ R^{p×p} are smooth coefficient matrices approximated using 
a sieve basis (B-splines or polynomial basis).

Feature Construction:
--------------------
For each lag j, compute discriminative features based on matrix norms:

    D(j) = sup_{t1,t2 ∈ [0,1]} ||Â_j(t1) - Â_j(t2)||

Aggregate features across upper lags:

    S = sup_{j ∈ upper-lag range} D(j)

Classification:
--------------
1. Compute per-series features S_k
2. Compute class medians S̄_1, S̄_2
3. Find threshold τ* that maximizes training accuracy
4. Classify new series based on distance to class medians

Robustness Features:
-------------------
- Handles unequal time series lengths via normalized time indices
- Robust to small/imbalanced samples via median-based features
- Stable under weak nonstationarity via regularization
"""

from __future__ import annotations

from typing import Literal, Tuple
import warnings

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class TimeVaryingMVAR:
    """
    Time-varying Multivariate Autoregressive model with sieve basis expansion.
    
    Parameters
    ----------
    order : int
        AR model order (lag b)
    n_basis : int
        Number of basis functions for time-varying coefficients
    basis_type : {'bspline', 'polynomial'}
        Type of sieve basis
    regularization : float
        Ridge regularization parameter (0 = OLS)
    degree : int
        Degree for basis functions (B-spline degree or polynomial degree)
    """
    
    def __init__(
        self,
        order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        degree: int = 3,
    ):
        self.order = order
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.regularization = regularization
        self.degree = degree
        
        self.n_channels = None
        self.coefficients_ = None  # Shape: (order, p, p, n_basis)
        self.scaler_ = StandardScaler()
        
    def _construct_basis(self, t: np.ndarray) -> np.ndarray:
        """
        Construct basis matrix Φ(t) ∈ R^{T × K} where K = n_basis.
        
        Parameters
        ----------
        t : array-like, shape (T,)
            Normalized time indices in [0, 1]
            
        Returns
        -------
        basis : ndarray, shape (T, n_basis)
            Basis function evaluations
        """
        t = np.asarray(t).ravel()
        n_time = len(t)
        
        if self.basis_type == 'bspline':
            # Construct B-spline basis with uniform knots
            knots = np.linspace(0, 1, self.n_basis - self.degree + 1)
            # Augment knots for full B-spline basis
            full_knots = np.concatenate([
                [0] * self.degree,
                knots,
                [1] * self.degree
            ])
            
            basis = np.zeros((n_time, self.n_basis))
            for i in range(self.n_basis):
                # Construct B-spline coefficients (indicator for basis i)
                c = np.zeros(self.n_basis)
                c[i] = 1.0
                spline = BSpline(full_knots, c, self.degree, extrapolate=False)
                basis[:, i] = spline(t)
                
        elif self.basis_type == 'polynomial':
            # Legendre-style orthogonal polynomial basis
            basis = np.zeros((n_time, self.n_basis))
            for k in range(self.n_basis):
                basis[:, k] = np.power(2 * t - 1, k)
                
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")
            
        return basis
    
    def _construct_design_matrix(
        self, 
        Z: np.ndarray, 
        basis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct design matrix for MVAR estimation.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p)
            Multivariate time series
        basis : ndarray, shape (T, n_basis)
            Basis evaluations at each time point
            
        Returns
        -------
        X_design : ndarray, shape (T - order, p * order * n_basis)
            Design matrix with lagged values weighted by basis functions
        y : ndarray, shape (T - order, p)
            Response variables
        """
        T, p = Z.shape
        T_eff = T - self.order
        n_features = p * self.order * self.n_basis
        
        X_design = np.zeros((T_eff, n_features))
        y = Z[self.order:]
        
        # For each time t in [order, T)
        for t_idx in range(T_eff):
            t = t_idx + self.order
            
            # Stack lagged values and basis functions
            feature_idx = 0
            for lag_j in range(1, self.order + 1):
                # Z_{t-j} ∈ R^p
                z_lag = Z[t - lag_j]
                
                # Basis functions at time t: Φ(t) ∈ R^K
                phi_t = basis[t]
                
                # Outer product: z_lag ⊗ phi_t ∈ R^{p * K}
                for ch in range(p):
                    for k in range(self.n_basis):
                        X_design[t_idx, feature_idx] = z_lag[ch] * phi_t[k]
                        feature_idx += 1
                        
        return X_design, y
    
    def fit(self, Z: np.ndarray) -> TimeVaryingMVAR:
        """
        Fit time-varying MVAR model via regularized least squares.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p)
            Multivariate time series (channels × time transposed to time × channels)
            
        Returns
        -------
        self : TimeVaryingMVAR
            Fitted model
        """
        if Z.ndim != 2:
            raise ValueError("Z must be 2D array (time × channels)")
            
        T, p = Z.shape
        self.n_channels = p
        
        # Normalize time indices
        t_norm = np.linspace(0, 1, T)
        
        # Construct basis
        basis = self._construct_basis(t_norm)
        
        # Standardize data (important for regularization)
        Z_scaled = self.scaler_.fit_transform(Z)
        
        # Construct design matrix
        X_design, y = self._construct_design_matrix(Z_scaled, basis)
        
        # Fit coefficients for each output channel separately
        self.coefficients_ = np.zeros((self.order, p, p, self.n_basis))
        
        for out_ch in range(p):
            if self.regularization > 0:
                ridge = Ridge(alpha=self.regularization, fit_intercept=False)
                ridge.fit(X_design, y[:, out_ch])
                theta = ridge.coef_
            else:
                # OLS solution
                theta = linalg.lstsq(X_design, y[:, out_ch])[0]
            
            # Reshape theta back to (order, p, n_basis)
            theta_reshaped = theta.reshape(self.order, p, self.n_basis)
            self.coefficients_[:, out_ch, :, :] = theta_reshaped
            
        return self
    
    def get_coefficient_matrices(self, t_eval: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate coefficient matrices A_j(t) at specified time points.
        
        Parameters
        ----------
        t_eval : array-like, shape (M,), optional
            Normalized time points in [0, 1] for evaluation.
            If None, uses uniform grid of 100 points.
            
        Returns
        -------
        A_matrices : ndarray, shape (M, order, p, p)
            Coefficient matrices A_j(t) for j=1,...,order and times in t_eval
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted first")
            
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)
        else:
            t_eval = np.asarray(t_eval).ravel()
            
        M = len(t_eval)
        basis = self._construct_basis(t_eval)  # Shape: (M, n_basis)
        
        A_matrices = np.zeros((M, self.order, self.n_channels, self.n_channels))
        
        # A_j(t) = Σ_k c_{j,k} φ_k(t)
        for m in range(M):
            for j in range(self.order):
                # Matrix multiply: coefficients[:, :, :, k] @ basis[m, k]
                A_matrices[m, j] = np.einsum(
                    'ijk,k->ij',
                    self.coefficients_[j],  # (p, p, n_basis)
                    basis[m]  # (n_basis,)
                )
                
        return A_matrices


class MVARFeatureExtractor:
    """
    Extract discriminative features from time-varying MVAR coefficients.
    
    Features are based on temporal variation in coefficient matrices:
        D(j) = sup_{t1,t2} ||A_j(t1) - A_j(t2)||
        
    Aggregated across upper lags:
        S = sup_{j ∈ upper_lags} D(j)
    """
    
    def __init__(
        self,
        mvar_order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        upper_lag_range: Tuple[int, int] | None = None,
        norm_type: Literal['fro', 'spectral', 'operator'] = 'fro',
        n_time_points: int = 50,
    ):
        """
        Parameters
        ----------
        mvar_order : int
            MVAR model order
        n_basis : int
            Number of basis functions
        basis_type : {'bspline', 'polynomial'}
            Sieve basis type
        regularization : float
            Ridge regularization strength
        upper_lag_range : tuple of int, optional
            (start, end) indices for upper lag aggregation.
            If None, uses last half of lags.
        norm_type : {'fro', 'spectral', 'operator'}
            Matrix norm for computing D(j)
        n_time_points : int
            Number of time points for evaluating supremum
        """
        self.mvar_order = mvar_order
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.regularization = regularization
        self.upper_lag_range = upper_lag_range
        self.norm_type = norm_type
        self.n_time_points = n_time_points
        
    def _matrix_norm(self, A: np.ndarray) -> float:
        """Compute specified matrix norm."""
        if self.norm_type == 'fro':
            return np.linalg.norm(A, ord='fro')
        elif self.norm_type == 'spectral':
            return np.linalg.norm(A, ord=2)
        elif self.norm_type == 'operator':
            # Operator norm = largest singular value
            return np.linalg.norm(A, ord=2)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def _compute_lag_variation(self, A_matrices: np.ndarray, lag_idx: int) -> float:
        """
        Compute D(j) = sup_{t1,t2} ||A_j(t1) - A_j(t2)||.
        
        Parameters
        ----------
        A_matrices : ndarray, shape (M, order, p, p)
            Coefficient matrices at M time points
        lag_idx : int
            Lag index j
            
        Returns
        -------
        d_j : float
            Maximum variation across time for lag j
        """
        M = A_matrices.shape[0]
        A_j = A_matrices[:, lag_idx]  # Shape: (M, p, p)
        
        # Compute all pairwise differences and their norms
        max_diff = 0.0
        for i in range(M):
            for j in range(i + 1, M):
                diff_norm = self._matrix_norm(A_j[i] - A_j[j])
                max_diff = max(max_diff, diff_norm)
                
        return max_diff
    
    def extract_features(self, Z: np.ndarray) -> float:
        """
        Extract scalar feature S from multivariate time series.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p) or (p, T)
            Multivariate time series
            
        Returns
        -------
        S : float
            Aggregated discriminative feature
        """
        # Ensure Z is (T, p)
        if Z.shape[0] < Z.shape[1]:
            Z = Z.T
            
        # Fit MVAR model
        mvar = TimeVaryingMVAR(
            order=self.mvar_order,
            n_basis=self.n_basis,
            basis_type=self.basis_type,
            regularization=self.regularization,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mvar.fit(Z)
        
        # Get coefficient matrices at evaluation points
        t_eval = np.linspace(0, 1, self.n_time_points)
        A_matrices = mvar.get_coefficient_matrices(t_eval)
        
        # Determine upper lag range
        if self.upper_lag_range is None:
            # Use last half of lags
            start = self.mvar_order // 2
            end = self.mvar_order
        else:
            start, end = self.upper_lag_range
            
        # Compute D(j) for each lag in upper range
        lag_variations = []
        for j in range(start, end):
            d_j = self._compute_lag_variation(A_matrices, j)
            lag_variations.append(d_j)
            
        # Aggregate: S = sup_j D(j)
        S = np.max(lag_variations) if lag_variations else 0.0
        
        return S


class MVARBinaryClassifier:
    """
    Binary classifier for multivariate time series based on MVAR features.
    
    Classification Algorithm:
    1. Extract feature S_k for each training series
    2. Compute class medians μ̂_1, μ̂_2
    3. Find threshold τ* maximizing training accuracy
    4. Classify new series: ŷ = argmin_c |S - μ̂_c|
    """
    
    def __init__(
        self,
        mvar_order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        upper_lag_range: Tuple[int, int] | None = None,
        norm_type: Literal['fro', 'spectral', 'operator'] = 'fro',
        n_time_points: int = 50,
        n_grid_points: int = 100,
    ):
        """
        Parameters
        ----------
        mvar_order : int
            MVAR model order
        n_basis : int
            Number of basis functions
        basis_type : {'bspline', 'polynomial'}
            Sieve basis type
        regularization : float
            Ridge regularization for stability
        upper_lag_range : tuple, optional
            Lag range for feature aggregation
        norm_type : {'fro', 'spectral', 'operator'}
            Matrix norm type
        n_time_points : int
            Time resolution for supremum computation
        n_grid_points : int
            Grid resolution for threshold search
        """
        self.feature_extractor = MVARFeatureExtractor(
            mvar_order=mvar_order,
            n_basis=n_basis,
            basis_type=basis_type,
            regularization=regularization,
            upper_lag_range=upper_lag_range,
            norm_type=norm_type,
            n_time_points=n_time_points,
        )
        self.n_grid_points = n_grid_points
        
        # Learned parameters
        self.class_medians_ = None  # {0: μ̂_0, 1: μ̂_1}
        self.threshold_ = None
        self.train_features_ = None
        self.train_labels_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> MVARBinaryClassifier:
        """
        Fit classifier on training data.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, T) or (n_samples, T, p)
            Training time series
        y : ndarray, shape (n_samples,)
            Binary labels {0, 1}
            
        Returns
        -------
        self : MVARBinaryClassifier
        """
        n_samples = X.shape[0]
        
        # Extract features for all training series
        print("Extracting MVAR features from training data...")
        features = np.zeros(n_samples)
        for i in range(n_samples):
            features[i] = self.feature_extractor.extract_features(X[i])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} series")
                
        self.train_features_ = features
        self.train_labels_ = y
        
        # Compute class medians (robust to outliers)
        self.class_medians_ = {
            0: np.median(features[y == 0]),
            1: np.median(features[y == 1]),
        }
        
        print(f"\nClass medians: {self.class_medians_}")
        
        # Find optimal threshold via grid search
        feature_min = features.min()
        feature_max = features.max()
        thresholds = np.linspace(feature_min, feature_max, self.n_grid_points)
        
        best_acc = 0.0
        best_thresh = None
        
        for thresh in thresholds:
            # Classify based on distance to class medians
            pred = np.where(
                np.abs(features - self.class_medians_[0]) < 
                np.abs(features - self.class_medians_[1]),
                0, 1
            )
            acc = np.mean(pred == y)
            
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                
        self.threshold_ = best_thresh
        
        print(f"Optimal threshold: {self.threshold_:.4f}")
        print(f"Training accuracy: {best_acc:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for new time series.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, T) or (n_samples, T, p)
            Test time series
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted binary labels
        """
        if self.class_medians_ is None:
            raise ValueError("Classifier must be fitted first")
            
        n_samples = X.shape[0]
        features = np.zeros(n_samples)
        
        print(f"Extracting features from {n_samples} test series...")
        for i in range(n_samples):
            features[i] = self.feature_extractor.extract_features(X[i])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} series")
                
        # Classify based on distance to class medians
        y_pred = np.where(
            np.abs(features - self.class_medians_[0]) < 
            np.abs(features - self.class_medians_[1]),
            0, 1
        )
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (soft assignment based on distance).
        
        Parameters
        ----------
        X : ndarray
            Test time series
            
        Returns
        -------
        proba : ndarray, shape (n_samples, 2)
            Class probabilities
        """
        if self.class_medians_ is None:
            raise ValueError("Classifier must be fitted first")
            
        n_samples = X.shape[0]
        features = np.zeros(n_samples)
        
        for i in range(n_samples):
            features[i] = self.feature_extractor.extract_features(X[i])
            
        # Distance-based soft assignment
        dist_0 = np.abs(features - self.class_medians_[0])
        dist_1 = np.abs(features - self.class_medians_[1])
        
        # Convert to probabilities via softmax of negative distances
        total_dist = dist_0 + dist_1
        proba = np.zeros((n_samples, 2))
        proba[:, 0] = dist_1 / (total_dist + 1e-10)
        proba[:, 1] = dist_0 / (total_dist + 1e-10)
        
        return proba

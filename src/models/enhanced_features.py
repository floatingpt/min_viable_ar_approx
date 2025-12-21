"""
Enhanced MVAR feature extraction with multi-dimensional feature vectors.

Replaces scalar feature S with a comprehensive feature vector including:
- Maximum temporal variation (S_max)
- Mean temporal variation (S_mean)
- Standard deviation of variation (S_std)
- Frobenius and spectral norm variants
- Per-lag statistics

Total: 5-10 features per window for improved discrimination.
"""

from __future__ import annotations

from typing import Literal, Tuple
import warnings

import numpy as np

from src.models.mvar_classifier import TimeVaryingMVAR


class EnhancedMVARFeatureExtractor:
    """
    Extract multi-dimensional feature vectors from time-varying MVAR coefficients.
    
    Features capture temporal variation in coefficient matrices across multiple
    perspectives to provide richer discriminative information than a single scalar.
    """
    
    def __init__(
        self,
        mvar_order: int = 3,
        n_basis: int = 10,
        basis_type: Literal['bspline', 'polynomial'] = 'bspline',
        regularization: float = 0.1,
        upper_lag_range: Tuple[int, int] | None = None,
        n_time_points: int = 50,
        include_per_lag: bool = False,
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
        n_time_points : int
            Number of time points for evaluating temporal variation
        include_per_lag : bool
            If True, include per-lag features (increases dimensionality)
        """
        self.mvar_order = mvar_order
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.regularization = regularization
        self.upper_lag_range = upper_lag_range
        self.n_time_points = n_time_points
        self.include_per_lag = include_per_lag
        
        # Cache for feature names
        self._feature_names = None
        
    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        if self._feature_names is not None:
            return self._feature_names
            
        names = [
            'S_max_fro',      # Maximum variation (Frobenius norm)
            'S_mean_fro',     # Mean variation (Frobenius norm)
            'S_std_fro',      # Std deviation of variation (Frobenius)
            'S_max_spec',     # Maximum variation (spectral norm)
            'S_mean_spec',    # Mean variation (spectral norm)
            'S_std_spec',     # Std deviation of variation (spectral)
        ]
        
        if self.include_per_lag:
            # Determine lag range
            if self.upper_lag_range is None:
                start = self.mvar_order // 2
                end = self.mvar_order
            else:
                start, end = self.upper_lag_range
                
            # Add per-lag features
            for j in range(start, end):
                names.append(f'D_lag{j+1}_fro')
                
        self._feature_names = names
        return names
        
    def _matrix_norm(self, A: np.ndarray, norm_type: str) -> float:
        """Compute specified matrix norm."""
        if norm_type == 'fro':
            return np.linalg.norm(A, ord='fro')
        elif norm_type in ('spectral', 'spec'):
            return np.linalg.norm(A, ord=2)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    
    def _compute_lag_variation(
        self,
        A_matrices: np.ndarray,
        lag_idx: int,
        norm_type: str = 'fro'
    ) -> float:
        """
        Compute D(j) = sup_{t1,t2} ||A_j(t1) - A_j(t2)||.
        
        Parameters
        ----------
        A_matrices : ndarray, shape (M, order, p, p)
            Coefficient matrices at M time points
        lag_idx : int
            Lag index j
        norm_type : str
            'fro' or 'spectral'
            
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
                diff_norm = self._matrix_norm(A_j[i] - A_j[j], norm_type)
                max_diff = max(max_diff, diff_norm)
                
        return max_diff
    
    def extract_features(self, Z: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from multivariate time series.
        
        Parameters
        ----------
        Z : ndarray, shape (T, p) or (p, T)
            Multivariate time series
            
        Returns
        -------
        features : ndarray, shape (n_features,)
            Feature vector containing:
            - S_max_fro: Maximum variation (Frobenius)
            - S_mean_fro: Mean variation (Frobenius)
            - S_std_fro: Std of variation (Frobenius)
            - S_max_spec: Maximum variation (spectral)
            - S_mean_spec: Mean variation (spectral)
            - S_std_spec: Std of variation (spectral)
            - [Optional] Per-lag variations D(j)
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
            start = self.mvar_order // 2
            end = self.mvar_order
        else:
            start, end = self.upper_lag_range
            
        # Compute D(j) for each lag with both norm types
        lag_variations_fro = []
        lag_variations_spec = []
        
        for j in range(start, end):
            d_j_fro = self._compute_lag_variation(A_matrices, j, 'fro')
            d_j_spec = self._compute_lag_variation(A_matrices, j, 'spectral')
            lag_variations_fro.append(d_j_fro)
            lag_variations_spec.append(d_j_spec)
            
        # Convert to arrays
        lag_variations_fro = np.array(lag_variations_fro)
        lag_variations_spec = np.array(lag_variations_spec)
        
        # Compute aggregate statistics
        features = []
        
        # Frobenius norm features
        features.append(np.max(lag_variations_fro) if len(lag_variations_fro) > 0 else 0.0)  # S_max_fro
        features.append(np.mean(lag_variations_fro) if len(lag_variations_fro) > 0 else 0.0)  # S_mean_fro
        features.append(np.std(lag_variations_fro) if len(lag_variations_fro) > 0 else 0.0)   # S_std_fro
        
        # Spectral norm features
        features.append(np.max(lag_variations_spec) if len(lag_variations_spec) > 0 else 0.0)  # S_max_spec
        features.append(np.mean(lag_variations_spec) if len(lag_variations_spec) > 0 else 0.0)  # S_mean_spec
        features.append(np.std(lag_variations_spec) if len(lag_variations_spec) > 0 else 0.0)   # S_std_spec
        
        # Optional: Include per-lag features
        if self.include_per_lag:
            features.extend(lag_variations_fro.tolist())
            
        return np.array(features)
    
    def transform(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Extract features from multiple time series.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, T) or (n_samples, T, p)
            Multiple time series
        verbose : bool
            If True, print progress
            
        Returns
        -------
        features : ndarray, shape (n_samples, n_features)
            Feature matrix
        """
        n_samples = X.shape[0]
        
        # Extract first sample to determine feature dimensionality
        first_features = self.extract_features(X[0])
        n_features = len(first_features)
        
        # Allocate feature matrix
        features = np.zeros((n_samples, n_features))
        features[0] = first_features
        
        # Extract remaining features
        if verbose:
            print(f"Extracting {n_features} MVAR features from {n_samples} samples...")
            
        for i in range(1, n_samples):
            features[i] = self.extract_features(X[i])
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")
                
        return features

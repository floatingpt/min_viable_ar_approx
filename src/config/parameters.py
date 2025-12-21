DEFAULT_PARAMS = {
    # Window extraction parameters
    'window_sec': 2.0,
    'stride_sec': 2.0,
    
    # Train/test split parameters
    'test_size': 0.2,
    'random_seed': 42,
    
    # Data paths (relative to project root)
    'dataset_root': 'siena-scalp-eeg-database-1.0.0',
    
    # MVAR model parameters
    'mvar_order': 3,
    'n_basis': 10,
    'basis_type': 'bspline',  # 'bspline' or 'polynomial'
    'regularization': 0.1,
    'upper_lag_range': None,  # None = last half of lags
    'norm_type': 'fro',  # 'fro', 'spectral', or 'operator'
    'n_time_points': 50,
    'n_grid_points': 100,
}

"""
Subject-level grouped stratified splitting to prevent data leakage.

This module ensures no windows from the same subject appear in both train and test sets.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def create_subject_grouped_split(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split grouped by subject ID to prevent data leakage.
    
    Parameters
    ----------
    subject_ids : ndarray, shape (n_samples,)
        Subject identifier for each window
    labels : ndarray, shape (n_samples,)
        Binary labels {0, 1} for each window
    test_size : float
        Proportion of subjects to hold out for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    train_idx : ndarray
        Indices of training samples
    test_idx : ndarray
        Indices of test samples
        
    Notes
    -----
    This function ensures that all windows from a given subject belong to
    either the training set OR the test set, never both. This prevents
    temporal/subject-specific information leakage.
    """
    # Use GroupShuffleSplit for subject-level splitting
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    # Get single train/test split
    train_idx, test_idx = next(gss.split(
        X=np.zeros(len(subject_ids)),  # Dummy X
        y=labels,
        groups=subject_ids
    ))
    
    return train_idx, test_idx


def verify_no_subject_leakage(
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> bool:
    """
    Verify that no subject appears in both train and test sets.
    
    Parameters
    ----------
    train_subjects : ndarray
        Subject IDs in training set
    test_subjects : ndarray
        Subject IDs in test set
        
    Returns
    -------
    is_valid : bool
        True if no overlap, False otherwise
    """
    train_set = set(train_subjects)
    test_set = set(test_subjects)
    overlap = train_set & test_set
    
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} subjects appear in both train and test!")
        print(f"   Overlapping subjects: {sorted(overlap)}")
        return False
    
    return True


def print_split_statistics(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Print detailed statistics about the train/test split."""
    train_subjects = subject_ids[train_idx]
    test_subjects = subject_ids[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    print("\n" + "=" * 80)
    print("SUBJECT-GROUPED SPLIT STATISTICS")
    print("=" * 80)
    
    # Subject counts
    n_unique_train = len(np.unique(train_subjects))
    n_unique_test = len(np.unique(test_subjects))
    print(f"Unique subjects:")
    print(f"  Train: {n_unique_train}")
    print(f"  Test:  {n_unique_test}")
    print(f"  Total: {n_unique_train + n_unique_test}")
    
    # Window counts
    print(f"\nWindow counts:")
    print(f"  Train: {len(train_idx)} windows")
    print(f"  Test:  {len(test_idx)} windows")
    print(f"  Total: {len(train_idx) + len(test_idx)} windows")
    
    # Class balance
    print(f"\nClass balance (Train):")
    train_class_counts = np.bincount(train_labels)
    print(f"  Class 0 (non-seizure): {train_class_counts[0]} ({100*train_class_counts[0]/len(train_labels):.1f}%)")
    print(f"  Class 1 (seizure):     {train_class_counts[1]} ({100*train_class_counts[1]/len(train_labels):.1f}%)")
    
    print(f"\nClass balance (Test):")
    test_class_counts = np.bincount(test_labels)
    print(f"  Class 0 (non-seizure): {test_class_counts[0]} ({100*test_class_counts[0]/len(test_labels):.1f}%)")
    print(f"  Class 1 (seizure):     {test_class_counts[1]} ({100*test_class_counts[1]/len(test_labels):.1f}%)")
    
    # Verify no leakage
    print(f"\nLeakage check:")
    is_valid = verify_no_subject_leakage(train_subjects, test_subjects)
    if is_valid:
        print("  ✓ No subject leakage detected")
    
    print("=" * 80)

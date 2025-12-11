from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.loader import load_har_features

def load_train_val_test(
    base_dir: str | Path = "data/raw/UCI HAR Dataset/UCI HAR Dataset",
    val_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load data and perform preprocessing pipeline.
    
    Steps:
    1. Load original train and test sets.
    2. Split validation set from the training set.
    3. Standardize features (fit on train, transform all).
    
    Parameters:
    ----------
    base_dir : str or Path
        Dataset directory.
    val_size : float
        Proportion of the training set to include in the validation split.
    random_state : int
        Random seed for reproducibility.
        
    Returns:
    -------
    Tuple containing 7 elements:
    (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    
    # 1. Load Data
    print("Loading raw data...")
    X_train_full, y_train_full = load_har_features(base_dir, split="train")
    X_test, y_test = load_har_features(base_dir, split="test")
    
    # 2. Train-Validation Split
    print(f"Splitting validation set (ratio: {val_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train_full
    )
    
    # 3. Standardization
    # Rule: Fit only on training data to avoid data leakage.
    print("Standardizing features...")
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("Preprocessing complete!")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

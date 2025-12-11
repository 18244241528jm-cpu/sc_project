from pathlib import Path
import numpy as np

def load_har_features(
    base_dir: str | Path = "data/raw/UCI HAR Dataset",
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load UCI HAR dataset (feature vectors).

    This function reads X (features) and y (labels) from text files into numpy arrays.

    Parameters:
    ----------
    base_dir : str or Path
        Root directory of the dataset.
    split : str
        "train" or "test".

    Returns:
    -------
    X : np.ndarray
        Feature matrix. Shape (n_samples, 561).
    y : np.ndarray
        Label vector. Shape (n_samples,).
    """
    
    base_dir = Path(base_dir)

    # Validate split argument
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split argument: {split}. Must be 'train' or 'test'.")

    # Construct file paths
    split_dir = base_dir / split
    
    X_path = split_dir / f"X_{split}.txt"
    y_path = split_dir / f"y_{split}.txt"

    # Check if files exist
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"File not found!\n"
            f"Checked paths:\n  {X_path}\n  {y_path}\n"
            f"Please ensure the dataset is located at {base_dir}."
        )

    print(f"Loading {split} data from {split_dir} ...")

    # Read files
    try:
        X = np.loadtxt(X_path)
    except ValueError as e:
        raise ValueError(f"Failed to read feature file: {X_path}\nOriginal error: {e}")
    
    try:
        y = np.loadtxt(y_path).astype(int)
    except ValueError as e:
        raise ValueError(f"Failed to read label file: {y_path}\nOriginal error: {e}")

    # Flatten y if necessary
    if y.ndim > 1:
        y = y.ravel()

    print(f"Done. X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def load_har_signals(
    base_dir: str | Path = "data/raw/UCI HAR Dataset",
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw Inertial Signals from the UCI HAR dataset.

    Parameters:
    ----------
    base_dir : str or Path
        Root directory of the dataset.
    split : str
        "train" or "test".

    Returns:
    -------
    signals : np.ndarray
        Raw signals 3D array.
        Shape: (n_samples, 128, 9)
        - 128: Time steps (2.56s @ 50Hz)
        - 9: Channels
    y : np.ndarray
        Label vector. Shape (n_samples,).
    """
    base_dir = Path(base_dir)
    
    # List of signal filenames. Order matters.
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    split_dir = base_dir / split
    signals_dir = split_dir / "Inertial Signals"
    
    if not signals_dir.exists():
        raise FileNotFoundError(f"Signal directory not found: {signals_dir}")
        
    # Load all 9 signal files
    loaded_signals = []
    print(f"Loading {split} raw signals from {signals_dir} ...")
    
    for name in signal_names:
        file_path = signals_dir / f"{name}_{split}.txt"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Missing signal file: {file_path}")
            
        # Read individual file (N, 128)
        try:
            signal = np.loadtxt(file_path)
        except ValueError as e:
             raise ValueError(f"Failed to read signal file: {file_path}\nOriginal error: {e}")
             
        loaded_signals.append(signal)
        
    # Stack them along the depth axis (3rd dimension)
    # Result shape: (N, 128, 9)
    signals = np.dstack(loaded_signals)
    
    # Load labels
    y_path = split_dir / f"y_{split}.txt"
    try:
        y = np.loadtxt(y_path).astype(int)
    except ValueError as e:
        raise ValueError(f"Failed to read label file: {y_path}\nOriginal error: {e}")

    if y.ndim > 1:
        y = y.ravel()
        
    print(f"Done. Signals shape: {signals.shape}, y shape: {y.shape}")
    return signals, y

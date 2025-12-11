import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure test can see parent directory
sys.path.append(str(Path(__file__).parent.parent))

from features import extract_features
from data.loader import load_har_features

def test_extract_features_shape():
    """Test if feature extraction output shape is correct."""
    # Fake data: 10 samples, 128 time steps, 9 channels
    N_SAMPLES = 10
    fake_signals = np.random.randn(N_SAMPLES, 128, 9)
    
    # Run extraction
    features = extract_features(fake_signals)
    
    # Verify shape
    # 9 channels * 7 metrics = 63 features
    assert features.shape == (N_SAMPLES, 63)

def test_extract_features_values():
    """Test if feature calculation values are correct."""
    # Fake data of all ones
    # Mean, Max, Min should all be 1.0
    fake_signals = np.ones((1, 128, 9))
    
    features = extract_features(fake_signals)
    
    # Check first feature (Mean of channel 0)
    mean_val = features[0, 0]
    
    assert np.isclose(mean_val, 1.0)

def test_feature_columns():
    """Test if expected number of statistics are calculated."""
    fake_signals = np.random.randn(1, 128, 9)
    features = extract_features(fake_signals)
    
    # 7 metrics: Mean, Std, Max, Min, Median, IQR, Energy
    # 9 channels
    expected_dim = 9 * 7
    assert features.shape[1] == expected_dim

def test_load_har_features_file_not_found():
    """Test if FileNotFoundError is raised when data directory is missing."""
    # Point to a non-existent directory
    fake_dir = Path("non_existent_data_folder")
    
    # Expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_har_features(base_dir=fake_dir, split="train")

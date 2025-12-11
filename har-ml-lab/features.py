import numpy as np
from scipy import stats

def extract_features(signals: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from raw inertial signals.

    This function calculates statistical metrics (Mean, Std, Max, Min, 
    Median, IQR, Energy) for each of the 9 channels in the input signals.

    Parameters:
    ----------
    signals : np.ndarray
        3D array of shape (N, 128, 9).
        - N: Number of samples
        - 128: Time steps per sample
        - 9: Number of channels (Body Acc xyz, Body Gyro xyz, Total Acc xyz)

    Returns:
    -------
    features : np.ndarray
        2D array of shape (N, n_features).
        n_features = 9 channels * 7 metrics = 63 features.
    """
    
    features_list = []
    
    # Process each sample individually
    N = signals.shape[0]
    print(f"Extracting features for {N} samples...")
    
    for i in range(N):
        # Get data for the i-th sample: shape (128, 9)
        sample_data = signals[i]
        
        sample_features = []
        
        # Extract features for each of the 9 channels
        for channel_idx in range(9):
            # Get time series for current channel: shape (128,)
            series = sample_data[:, channel_idx]
            
            # --- Feature Extraction ---
            
            # 1. Mean
            sample_features.append(np.mean(series))
            
            # 2. Standard Deviation
            sample_features.append(np.std(series))
            
            # 3. Max & Min
            sample_features.append(np.max(series))
            sample_features.append(np.min(series))
            
            # 4. Median
            sample_features.append(np.median(series))
            
            # 5. Interquartile Range (IQR)
            sample_features.append(stats.iqr(series))
            
            # 6. Energy (Mean of squared values)
            sample_features.append(np.mean(series**2))
            
            # --- End Extraction ---
            
        features_list.append(sample_features)
        
    features_matrix = np.array(features_list)
    print(f"Feature extraction complete. Shape: {features_matrix.shape}")
    
    return features_matrix

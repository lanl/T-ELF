from sklearn.model_selection import train_test_split
import numpy as np

def sample_matrix(X, sample_ratio=0.05, random_state=42, stratify=True):
    """
    Samples a specified ratio of entries from a binary matrix, maintaining the proportion of 0s and 1s.
    Sets the sampled indices to 0 in the original matrix.
   
    Parameters:
    - matrix (np.ndarray): Binary matrix of 0s and 1s.
    - sample_ratio (float): Proportion of entries to sample. Default is 0.1 (10%).
    - random_state (int): Seed for reproducibility. Default is 42.
   
    Returns:
    - np.ndarray: Matrix with sampled entries set to 0s.
    - np.ndarray: Array of sampled indices in the flattened matrix format.
    """
    # Flatten the matrix and create indices
    flat_matrix = X.flatten()
    indices = np.arange(flat_matrix.size)
 
    # Perform stratified sampling to maintain the 0/1 ratio
    if stratify:
        _, sampled_indices = train_test_split(
            indices,
            test_size=sample_ratio,  # Specify the sample ratio
            stratify=flat_matrix,  # Maintain the 0/1 ratio
            random_state=random_state
        )
    else:
        _, sampled_indices = train_test_split(
            indices,
            test_size=sample_ratio,  # Specify the sample ratio
            random_state=random_state
        )
    
    # Set sampled entries to NaN in the original matrix
    Xtrain = X.astype(float).copy()
    np.put(Xtrain, sampled_indices, np.nan)

    removed_coords = np.argwhere(np.isnan(Xtrain))
    MASK = np.ones(Xtrain.shape)

    for x,y in removed_coords:
        MASK[x,y] = 0
        Xtrain[x,y] = 0
   
    return X.astype("float32"), Xtrain.astype("float32"), MASK, removed_coords
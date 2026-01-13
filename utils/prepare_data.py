
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def min_max_normalize(train_array, save_factors_path):
    """
    Min-max normalize a 2D NumPy array by columns and save normalization factors.
    
    Parameters:
        train_array (np.ndarray): 2D NumPy array to normalize (shape: [n_samples, n_features]).
        save_factors_path (str): Path to save the normalization factors (min and max values).
    
    Returns:
        normalized_array (np.ndarray): Min-max normalized 2D NumPy array.
    """
    # Compute min and max for each column
    min_vals = np.min(train_array, axis=0)
    max_vals = np.max(train_array, axis=0)
    print(f"Min: {min_vals}")
    print(f"Max: {max_vals}")
    # Normalize the array
    normalized_array = (train_array - min_vals) / (max_vals - min_vals)
    
    # Save normalization factors (min and max values for each column)
    # normalization_factors = {"min": min_vals, "max": max_vals}
    # with open(save_factors_path, "wb") as f:
    #     pickle.dump(normalization_factors, f)
    
    return normalized_array, min_vals, max_vals



def std_normalize(train_array, save_factors_path):
    """
    Normalize a 2D NumPy array by columns and save normalization factors.
    
    Parameters:
        train_array (np.ndarray): 2D NumPy array to normalize (shape: [n_samples, n_features]).
        save_factors_path (str): Path to save the normalization factors.
    
    Returns:
        normalized_array (np.ndarray): Normalized 2D NumPy array.
    """
    # Compute mean and std for each column
    mean = np.mean(train_array, axis=0)
    std = np.std(train_array, axis=0)
    
    print(f"Mean: {mean}")
    print(f"STD: {std}")
    # Avoid division by zero
    std[std == 0] = 1
    
    # Normalize the array
    normalized_array = (train_array - mean) / std

    # Save normalization factors
    # normalization_factors = {"mean": mean, "std": std}
    # with open(save_factors_path, "wb") as f:
    #     pickle.dump(normalization_factors, f)
    
    return normalized_array, mean, std



def normalize_test_array(test_array, use_standardization, fac1, fac2, load_factors_path):
    """
    Normalize a 2D NumPy array using saved normalization factors (min-max or standardization).
    
    Parameters:
        test_array (np.ndarray): 2D NumPy array to normalize (shape: [n_samples, n_features]).
        load_factors_path (str): Path to load the normalization factors (min, max, mean, and std values).
        use_standardization (bool): Whether to apply standardization (True) or min-max normalization (False).
    
    Returns:
        normalized_array (np.ndarray): Normalized 2D NumPy array.
    """
    # Load normalization factors
    # with open(load_factors_path, "rb") as f:
    #     normalization_factors = pickle.load(f)
    
    # If applying min-max normalization
    if not use_standardization:
        max_vals = fac1 #normalization_factors["max"]
        min_vals = fac2 #normalization_factors["min"]
        # Normalize the test data using the saved min and max values (min-max normalization)
        normalized_array = (test_array - min_vals) / (max_vals - min_vals)
    
    # If applying standardization (z-score normalization)
    else:
        mean_vals = fac1 #normalization_factors["mean"]
        std_vals = fac2 #normalization_factors["std"]
        # Normalize the test data using the saved mean and std values (standardization)
        std_vals[std_vals == 0] = 1
        normalized_array = (test_array - mean_vals) / std_vals
    
    return normalized_array


def get_data_loader(distance_type, Xtrain, Xtest, Ytrain, Ytest, batch_size):
    
    if distance_type=='euc':
        train_x=Xtrain[:, 0:2]
        test_x=Xtest[:, 0:2]
    elif distance_type=='dtw':
        train_x=Xtrain[:, 2:]
        test_x=Xtest[:, 2:]
    elif distance_type=='both':
        train_x=Xtrain
        test_x=Xtest
    else:
        print("distance incorrect")

    print("\n ***** Normalization ***** \n")
    train_x_norm, means, stds = std_normalize(train_x, 'normalization_factors.pkl')
    # print(train_x_norm[:20])
    train_x_norm2, mins, maxs = min_max_normalize(train_x, 'normalization_factors.pkl')
    # print(train_x_norm2[:20])

    # Normalize test data using standardization (z-score normalization)
    test_x_norm = normalize_test_array(test_x, True, means, stds, 'normalization_factors.pkl')
    # Normalize test data using min max Normalization
    test_x_norm2 = normalize_test_array(test_x, False, maxs, mins, 'normalization_factors.pkl')

    # Convert numpy arrays to PyTorch tensors
    train_tensor = torch.tensor(train_x_norm, dtype=torch.float32)  # Input features
    Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for binary targets
    
    test_tensor = torch.tensor(test_x_norm, dtype=torch.float32)
    Ytest_tensor = torch.tensor(Ytest, dtype=torch.float32).unsqueeze(1)

    # Create TensorDatasets
    train_dataset1 = TensorDataset(train_tensor, Ytrain_tensor)
    test_dataset = TensorDataset(test_tensor, Ytest_tensor)

    # Define the split ratio (e.g., 90% for training, 10% for validation)
    train_size = int(0.9 * len(train_dataset1))  # 90% for training
    val_size = len(train_dataset1) - train_size  # 10% for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(train_dataset1, [train_size, val_size])
    

    # Create DataLoaders for batching

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
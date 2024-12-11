import torch
import numpy as np
from typing import List

    
def get_y_from_noise(z: torch.Tensor, idx_freq: torch.Tensor) -> torch.Tensor:
    idx_noise = normalize_to_standard_normal(z) # bsz, 
    idx_uniform = noise_to_uniform(idx_noise) # bsz, 
    y = uniform_to_custom_distribution_refine(idx_uniform, idx_freq)
    return y
    
def normalize_to_standard_normal(data: torch.Tensor) -> torch.Tensor:
    """
    Aggregate data with mean method and normalize to a standard normal distribution (mean=0, std=1).
    
    Args:
        data (torch.Tensor): Input tensor of shape (bsz, 4, 768).
    
    Returns:
        torch.Tensor: Tensor of shape (bsz,), normalized to standard normal.
    """
    # Step 1: Compute mean across the last two dimensions
    aggregated_mean = data.mean(dim=[1, 2, 3], keepdim=True)  # Shape: (bsz, 1)
    
    # Step 2: Compute theoretical standard deviation after mean aggregation
    # For a mean of n independent samples, std becomes original_std / sqrt(n)
    n = data.size(1) * data.size(2) * data.size(3)  # Total elements in the aggregated dimensions
    aggregated_std = 1 / torch.sqrt(torch.tensor(float(n)))
    
    # Step 3: Normalize the aggregated mean to standard normal distribution
    normalized_data = aggregated_mean / aggregated_std  # Adjust to standard normal (mean=0, std=1)
    
    return normalized_data[:,0,0]




def noise_to_uniform(noise: torch.Tensor, mu: float = 0, sigma: float = 1.) -> torch.Tensor:
    """
    Converts Gaussian noise to a uniform distribution within the range [0, 1].
    
    Args:
        noise (Tensor): Input Gaussian noise (arbitrary shape).
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
        Tensor: Tensor with values uniformly distributed in [0, 1].
    """
    # Convert Gaussian to standard uniform in [0, 1]
    uniform_data = 0.5 * (1 + torch.erf((noise - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))
    
    # Clamp values strictly within [0, 1) to ensure no out-of-bounds values
    # uniform_data = torch.clamp(uniform_data, 0, 1 - 1e-7)
    
    return uniform_data




def uniform_to_custom_distribution(uniform_data: List[float], frequency: List[int]) -> List[int]:
    """
    Convert a Uniform distribution [0, 1] into a custom frequency distribution.

    Args:
        uniform_data (list of float): List of values sampled from a uniform distribution [0, 1].
        frequency (list of int): Target frequency distribution. frequency[i] indicates the desired frequency for integer i.

    Returns:
        list: A list of integers sampled according to the custom distribution.
    """
    # Normalize frequency to get probabilities
    total_count = sum(frequency)
    probabilities = [freq / total_count for freq in frequency]
    
    # Compute cumulative distribution for mapping
    cumulative_prob = np.cumsum(probabilities)
    
    # Map uniform data to the custom distribution
    result = []
    for u in uniform_data:
        for i, cp in enumerate(cumulative_prob):
            if u <= cp:
                result.append(i)
                break

    return result


import bisect
def uniform_to_custom_distribution_refine(uniform_data: List[float], frequency: List[int]) -> List[int]:
    """
    Convert a Uniform distribution [0, 1] into a custom frequency distribution.

    Args:
        uniform_data (list of float): List of values sampled from a uniform distribution [0, 1].
        frequency (list of int): Target frequency distribution. frequency[i] indicates the desired frequency for integer i.

    Returns:
        list: A list of integers sampled according to the custom distribution.
    """
    # Normalize frequency to get probabilities
    total_count = sum(frequency)
    probabilities = np.array(frequency) / total_count
    
    # Compute cumulative distribution for mapping
    cumulative_prob = np.cumsum(probabilities)
    
    # Use binary search to map uniform data to the custom distribution
    result = [bisect.bisect_right(cumulative_prob, u) for u in uniform_data]
    
    return result


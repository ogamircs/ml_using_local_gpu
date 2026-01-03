import torch
import os


def generate_synthetic_data(n_samples, n_features, seed=42):
    """
    Generate synthetic data with correlated features for PCA testing.

    Args:
        n_samples: number of samples
        n_features: number of features
        seed: random seed for reproducibility

    Returns:
        X: (n_samples, n_features) tensor
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, n_features, dtype=torch.float32)

    # Add correlated features to make PCA meaningful
    for i in range(0, n_features - 10, 10):
        X[:, i+1:i+5] = X[:, i:i+1] * 0.8 + torch.randn(n_samples, 4) * 0.2

    return X


def save_data(X, filepath):
    """Save tensor to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert to numpy and save
    import numpy as np
    np.savetxt(filepath, X.numpy(), delimiter=',', fmt='%.6f')

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved data to {filepath} ({size_mb:.2f} MB)")


def load_data(filepath):
    """Load tensor from CSV file."""
    import numpy as np
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    return torch.from_numpy(data)


def get_data_size_mb(n_samples, n_features):
    """Calculate data size in MB for float32."""
    return (n_samples * n_features * 4) / (1024 * 1024)

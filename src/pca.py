import torch


def pca_torch(X, n_components):
    """
    Perform PCA using SVD on the given tensor.

    Args:
        X: (n_samples, n_features) tensor
        n_components: number of principal components to keep

    Returns:
        X_transformed: projected data
        components: principal components
        explained_variance_ratio: variance explained by each component
    """
    # Center the data
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Perform SVD
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # Get principal components
    components = Vt[:n_components]

    # Project data onto principal components
    X_transformed = X_centered @ components.T

    # Calculate explained variance
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return X_transformed, components, explained_variance_ratio[:n_components]

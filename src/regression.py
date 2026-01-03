import torch


def linear_regression_matrix(X, y):
    """
    Perform linear regression using the normal equation (matrix form).

    β = (X^T X)^(-1) X^T y

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) or (n_samples, n_targets) target values

    Returns:
        coefficients: (n_features,) or (n_features, n_targets) regression coefficients
        predictions: predicted values
        r_squared: coefficient of determination
    """
    # Add bias term (column of ones)
    n_samples = X.shape[0]
    ones = torch.ones(n_samples, 1, dtype=X.dtype, device=X.device)
    X_bias = torch.cat([ones, X], dim=1)

    # Normal equation: β = (X^T X)^(-1) X^T y
    # Using torch.linalg.lstsq for numerical stability
    result = torch.linalg.lstsq(X_bias, y.unsqueeze(1) if y.dim() == 1 else y)
    coefficients = result.solution.squeeze()

    # Predictions
    predictions = X_bias @ coefficients

    # R-squared
    ss_res = ((y - predictions) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    return coefficients, predictions, r_squared


def linear_regression_direct(X, y):
    """
    Direct implementation using explicit matrix inverse.
    Less stable but demonstrates the math clearly.

    β = (X^T X)^(-1) X^T y
    """
    n_samples = X.shape[0]
    ones = torch.ones(n_samples, 1, dtype=X.dtype, device=X.device)
    X_bias = torch.cat([ones, X], dim=1)

    # Explicit normal equation
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y

    # Solve using Cholesky decomposition (faster for symmetric positive definite)
    L = torch.linalg.cholesky(XtX)
    coefficients = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze()

    predictions = X_bias @ coefficients

    ss_res = ((y - predictions) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    return coefficients, predictions, r_squared


def generate_regression_data(n_samples, n_features, noise=0.1, seed=42):
    """
    Generate synthetic regression data with known coefficients.

    Args:
        n_samples: number of samples
        n_features: number of features
        noise: noise level (std deviation)
        seed: random seed

    Returns:
        X: feature matrix
        y: target values
        true_coefficients: the actual coefficients used to generate data
    """
    torch.manual_seed(seed)

    # Generate features
    X = torch.randn(n_samples, n_features, dtype=torch.float32)

    # Generate true coefficients (including bias)
    true_coefficients = torch.randn(n_features + 1, dtype=torch.float32)

    # Generate targets: y = bias + X @ coef + noise
    ones = torch.ones(n_samples, 1, dtype=torch.float32)
    X_bias = torch.cat([ones, X], dim=1)
    y = X_bias @ true_coefficients + torch.randn(n_samples) * noise

    return X, y, true_coefficients

#!/usr/bin/env python
"""
GPU ML Benchmark

Compares ML algorithm performance between CPU and GPU using PyTorch.
Supports PCA (via SVD) and Linear Regression (via matrix form).
"""

import torch
import time
import argparse

from src import pca_torch, generate_synthetic_data, save_data, load_data
from src import linear_regression_matrix, linear_regression_direct, generate_regression_data
from src.data_generator import get_data_size_mb


def warmup_gpu():
    """Warm up GPU with a small computation."""
    print("Warming up GPU...")
    dummy = torch.randn(1000, 1000, device='cuda')
    _ = torch.linalg.svd(dummy)
    torch.cuda.synchronize()
    del dummy
    torch.cuda.empty_cache()


def run_pca_benchmark(X, n_components, device):
    """Run PCA on the specified device and return timing."""
    X_device = X.to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    X_transformed, components, var_ratio = pca_torch(X_device, n_components)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        'transformed': X_transformed,
        'components': components,
        'var_ratio': var_ratio,
        'time': elapsed
    }


def run_regression_benchmark(X, y, device, method='lstsq'):
    """Run linear regression on the specified device and return timing."""
    X_device = X.to(device)
    y_device = y.to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()

    if method == 'lstsq':
        coefficients, predictions, r_squared = linear_regression_matrix(X_device, y_device)
    else:
        coefficients, predictions, r_squared = linear_regression_direct(X_device, y_device)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return {
        'coefficients': coefficients,
        'predictions': predictions,
        'r_squared': r_squared,
        'time': elapsed
    }


def benchmark_pca(args):
    """Run PCA benchmark."""
    print("=" * 60)
    print("GPU PCA Benchmark")
    print("=" * 60)

    # Load or generate data
    if args.load_data:
        print(f"Loading data from {args.load_data}...")
        X = load_data(args.load_data)
        n_samples, n_features = X.shape
    else:
        n_samples = args.samples
        n_features = args.features
        size_mb = get_data_size_mb(n_samples, n_features)

        print(f"Generating synthetic data...")
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features:,}")
        print(f"  Size: {size_mb:.2f} MB")

        X = generate_synthetic_data(n_samples, n_features, seed=args.seed)

        if args.save_data:
            filepath = f"data/pca_{n_samples}x{n_features}.csv"
            save_data(X, filepath)

    print()
    n_components = min(args.components, min(X.shape))

    warmup_gpu()
    print()

    # Run benchmarks
    print("-" * 60)
    print("Running PCA on CPU...")
    cpu_result = run_pca_benchmark(X, n_components, torch.device('cpu'))
    print(f"  Time: {cpu_result['time']:.4f} seconds")
    print(f"  Explained variance ({n_components} components): {cpu_result['var_ratio'].sum().item():.4f}")
    print()

    print("Running PCA on GPU...")
    gpu_result = run_pca_benchmark(X, n_components, torch.device('cuda'))
    print(f"  Time: {gpu_result['time']:.4f} seconds")
    print(f"  Explained variance ({n_components} components): {gpu_result['var_ratio'].sum().item():.4f}")
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Data size:       {n_samples:,} x {n_features:,} ({get_data_size_mb(n_samples, n_features):.2f} MB)")
    print(f"Components:      {n_components}")
    print(f"CPU time:        {cpu_result['time']:.4f} seconds")
    print(f"GPU time:        {gpu_result['time']:.4f} seconds")
    print(f"Speedup:         {cpu_result['time'] / gpu_result['time']:.2f}x")
    print(f"GPU memory:      {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print("=" * 60)


def benchmark_regression(args):
    """Run regression benchmark."""
    print("=" * 60)
    print("GPU Linear Regression Benchmark (Matrix Form)")
    print("=" * 60)

    n_samples = args.samples
    n_features = args.features
    size_mb = get_data_size_mb(n_samples, n_features)

    print(f"Generating synthetic regression data...")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features:,}")
    print(f"  Size: {size_mb:.2f} MB")

    X, y, true_coef = generate_regression_data(n_samples, n_features, noise=0.1, seed=args.seed)

    if args.save_data:
        filepath = f"data/regression_{n_samples}x{n_features}.csv"
        # Save X and y together
        import numpy as np
        data = torch.cat([X, y.unsqueeze(1)], dim=1)
        save_data(data, filepath)

    print()
    warmup_gpu()
    print()

    # Run benchmarks with lstsq method
    print("-" * 60)
    print("Method: Least Squares (torch.linalg.lstsq)")
    print("-" * 60)

    print("Running regression on CPU...")
    cpu_result = run_regression_benchmark(X, y, torch.device('cpu'), method='lstsq')
    print(f"  Time: {cpu_result['time']:.4f} seconds")
    print(f"  R-squared: {cpu_result['r_squared'].item():.6f}")
    print()

    print("Running regression on GPU...")
    gpu_result = run_regression_benchmark(X, y, torch.device('cuda'), method='lstsq')
    print(f"  Time: {gpu_result['time']:.4f} seconds")
    print(f"  R-squared: {gpu_result['r_squared'].item():.6f}")
    print()

    # Run benchmarks with direct Cholesky method
    print("-" * 60)
    print("Method: Direct (Cholesky decomposition)")
    print("-" * 60)

    print("Running regression on CPU...")
    cpu_direct = run_regression_benchmark(X, y, torch.device('cpu'), method='direct')
    print(f"  Time: {cpu_direct['time']:.4f} seconds")
    print(f"  R-squared: {cpu_direct['r_squared'].item():.6f}")
    print()

    print("Running regression on GPU...")
    gpu_direct = run_regression_benchmark(X, y, torch.device('cuda'), method='direct')
    print(f"  Time: {gpu_direct['time']:.4f} seconds")
    print(f"  R-squared: {gpu_direct['r_squared'].item():.6f}")
    print()

    # Coefficient accuracy
    coef_error = torch.abs(cpu_result['coefficients'] - true_coef).mean()
    print(f"Mean coefficient error: {coef_error.item():.6f}")
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Data size:            {n_samples:,} x {n_features:,} ({size_mb:.2f} MB)")
    print()
    print("Least Squares Method:")
    print(f"  CPU time:           {cpu_result['time']:.4f} seconds")
    print(f"  GPU time:           {gpu_result['time']:.4f} seconds")
    print(f"  Speedup:            {cpu_result['time'] / gpu_result['time']:.2f}x")
    print()
    print("Cholesky Method:")
    print(f"  CPU time:           {cpu_direct['time']:.4f} seconds")
    print(f"  GPU time:           {gpu_direct['time']:.4f} seconds")
    print(f"  Speedup:            {cpu_direct['time'] / gpu_direct['time']:.2f}x")
    print()
    print(f"GPU memory:           {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='GPU ML Benchmark')
    parser.add_argument('--mode', type=str, default='pca', choices=['pca', 'regression', 'all'],
                        help='Benchmark mode: pca, regression, or all (default: pca)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples (default: 10000)')
    parser.add_argument('--features', type=int, default=2621,
                        help='Number of features (default: 2621 for ~100MB)')
    parser.add_argument('--components', type=int, default=50,
                        help='Number of PCA components (default: 50)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save generated data to data/ folder')
    parser.add_argument('--load-data', type=str, default=None,
                        help='Load data from CSV file (PCA mode only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    if args.mode == 'pca':
        benchmark_pca(args)
    elif args.mode == 'regression':
        benchmark_regression(args)
    else:  # all
        benchmark_pca(args)
        print("\n" * 2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        benchmark_regression(args)

    return 0


if __name__ == "__main__":
    exit(main())

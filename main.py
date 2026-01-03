#!/usr/bin/env python
"""
GPU PCA Benchmark

Compares PCA performance between CPU and GPU using PyTorch.
Generates synthetic data and runs PCA using SVD decomposition.
"""

import torch
import time
import argparse
import os

from src import pca_torch, generate_synthetic_data, save_data, load_data
from src.data_generator import get_data_size_mb


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

    return X_transformed, components, var_ratio, elapsed


def main():
    parser = argparse.ArgumentParser(description='GPU PCA Benchmark')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples (default: 10000)')
    parser.add_argument('--features', type=int, default=2621,
                        help='Number of features (default: 2621 for ~100MB)')
    parser.add_argument('--components', type=int, default=50,
                        help='Number of PCA components (default: 50)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save generated data to data/ folder')
    parser.add_argument('--load-data', type=str, default=None,
                        help='Load data from CSV file instead of generating')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("GPU PCA Benchmark")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

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
            filepath = f"data/synthetic_{n_samples}x{n_features}.csv"
            save_data(X, filepath)

    print()
    n_components = min(args.components, min(X.shape))

    # Warm up GPU
    print("Warming up GPU...")
    dummy = torch.randn(1000, 1000, device='cuda')
    _ = torch.linalg.svd(dummy)
    torch.cuda.synchronize()
    del dummy
    torch.cuda.empty_cache()
    print()

    # Run CPU benchmark
    print("-" * 60)
    print("Running PCA on CPU...")
    cpu_result = run_pca_benchmark(X, n_components, torch.device('cpu'))
    X_cpu, _, var_cpu, cpu_time = cpu_result
    print(f"  Time: {cpu_time:.4f} seconds")
    print(f"  Explained variance ({n_components} components): {var_cpu.sum().item():.4f}")
    print()

    # Run GPU benchmark
    print("Running PCA on GPU...")
    gpu_result = run_pca_benchmark(X, n_components, torch.device('cuda'))
    X_gpu, _, var_gpu, gpu_time = gpu_result
    print(f"  Time: {gpu_time:.4f} seconds")
    print(f"  Explained variance ({n_components} components): {var_gpu.sum().item():.4f}")
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Data size:       {n_samples:,} x {n_features:,} ({get_data_size_mb(n_samples, n_features):.2f} MB)")
    print(f"Components:      {n_components}")
    print(f"CPU time:        {cpu_time:.4f} seconds")
    print(f"GPU time:        {gpu_time:.4f} seconds")
    print(f"Speedup:         {cpu_time / gpu_time:.2f}x")
    print(f"GPU memory:      {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

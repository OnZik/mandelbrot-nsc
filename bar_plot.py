# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:54:08 2026

@author: zikan
"""

import matplotlib.pyplot as plt

# Configuration
N = 1024

# Benchmark results
# Replace the "..." with your actual runtimes from MP1/MP2
results = {
    "GPU f32": 0.003,
    "GPU f64": 0.026,
    "Naive": 7.264, 
    "NumPy": 1.016,   
    "Numba": 0.063,   
    "Multiprocessing": 0.017,
    "Dask Local": 0.135
}

N2 = 8192
results2 = {
    "GPU f32": 0.242,
    "GPU f64": 0.963,
    "Numba": 5.187,   
    "Multiprocessing": 0.667,
    "Dask Local": 1.369,
    "Dask Cluster": 1.648
}



def plot(N, r):
    # Data Extraction
    names, times = zip(*r.items())
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(names, times, log=True, color='skyblue', edgecolor='navy')
    
    # Labeling and Styling
    plt.ylabel("Seconds (log scale)")
    plt.title(f"Mandelbrot Benchmark (N={N})")
    plt.xticks(rotation=30, ha="right")
    
    # Finalize and Save
    plt.tight_layout()
    #plt.savefig("benchmark_mp3.png", dpi=150)
    plt.show()

plot(N, results)
plot(N2, results2)
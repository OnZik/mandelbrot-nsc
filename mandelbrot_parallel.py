# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:18 2026

@author: zikan
"""

import time, statistics

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import statistics, time, os

@njit (cache=True)
def mandelbrot_pixel(c_real, c_imag , max_iter):
    z_real = z_imag = 0.0
    for n in range(max_iter):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return n;
        new_z_imag = 2.0*z_real*z_imag + c_imag
        new_z_real = z_real*z_real - z_imag*z_imag + c_real
        
        z_real = new_z_real
        z_imag = new_z_imag
    return max_iter

@njit (cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / (N-1)
    dy = (y_max - y_min) / (N-1)
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel_p(x_min, x_max, y_min, y_max, N1, N2, max_iter, n_workers=12, n_chunks=None, pool=None):
    
    N = N1
    if n_chunks is None:
        n_chunks = n_workers
        
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    #tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    
    if pool == None:
        with Pool(processes=n_workers) as p:
        #p.map(_worker, tiny) # un-timed warm-up: Numba JIT in workers
            parts = p.map(_worker, chunks)
    
    else:
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)



def benchmark_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter):
    # warm-up
    mandelbrot_serial(8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")
    
    return t_serial

def mandelbrot_plot(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter):
    result = mandelbrot_parallel_p(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)    
    
    #--  PLOT --
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25], cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    
def sweep_workers(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter):
    # Sweep number of workers
    
    core_counts = []
    speedups = []
    
    t_serial = benchmark_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        
    for n_workers in range(1, os.cpu_count() + 1):
        tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
        
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        
        with Pool(processes=n_workers) as pool:
        # Warm-up: load JIT cache in workers
            pool.map(_worker, tiny) # warm-up: Numba JIT in all workers
            times = []
            
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        
        speedup = t_serial / t_par
        speedups.append(speedup)
        core_counts.append(n_workers)
        
        print(f"Parallel: {n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")      
    
    # Plot
    plt.figure()
    plt.plot(core_counts, speedups, marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Core Count")
    plt.grid()
    plt.show()
    
    
def benchmark_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks, n_workers):
    t_serial = benchmark_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
        row = row_end
    
    with Pool(processes=n_workers) as pool:
        # Warm-up: load JIT cache in workers
        pool.map(_worker, tiny) 
        
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            np.vstack(pool.map(_worker, chunks))
            times.append(time.perf_counter() - t0)
            
    t_par = statistics.median(times)
    speedup = t_serial / t_par
    print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x")  
    
    
def chunk_sweep(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers):
    # Chunk-count sweep (M2): one Pool per config    
    t_serial = benchmark_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    
    for mult in [1, 2, 3, 4, 8, 16]:
        n_chunks = mult * n_workers
        
        chunk_size = max(1, N // n_chunks)
        chunks, row = [], 0
        
        while row < N:
            row_end = min(row + chunk_size, N)
            chunks.append((row, row_end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = row_end
        
        with Pool(processes=n_workers) as pool:
            # Warm-up: load JIT cache in workers
            pool.map(_worker, tiny) 
            
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
                
        t_par = statistics.median(times)
        lif = (n_workers * t_par / t_serial) - 1
        
        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
    

if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.0, 1.0, -1.25, 1.25
    #n_chunks = 48
    n_workers = 12
    
    output = mandelbrot_parallel_p(X_MIN, X_MAX, Y_MIN, Y_MAX, N, N, max_iter)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(output, extent=[-2.5, 1.0, -1.25, 1.25], cmap='inferno', origin='lower', aspect='equal')
    
    
    # mandelbrot_plot(1024, -2.0, 1.0, -1.25, 1.25, 100)
    
    # sweep_workers(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    
    # chunk_sweep(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers)
      
    # Serial baseline (Numba already warm after M1 warm-up)
    # benchmark_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  
    
    #benchmark_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks, n_workers)
    
    
         
    
    
    


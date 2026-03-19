# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:18 2026

@author: zikan
"""


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
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
    max_iter=100, n_workers=4, n_chunks = None, pool = None):
    
    if n_chunks is None:
        n_chunks = n_workers
        
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
        
    if pool is not None:
        # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
        
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny) # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)
    
    return np.vstack(parts)

if __name__ == '__main__':
    result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, n_workers=4)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25], cmap='inferno', origin='lower', aspect='equal')
    
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    
    # --- MP2 M3: benchmark (in __main__ block) ---
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    # Serial baseline (Numba already warm after M1 warm-up)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    
    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks) # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
        
    
    best_workers = 12 

    configs = [1, 2, 4, 8, 16]
    
    print("\nChunk sweep (fixed workers):")
    print("chunks    time (s)    LIF")
    
    for factor in configs:
        n_chunks = factor * best_workers
    
        # Build chunks manually 
        chunk_size = max(1, N // n_chunks)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end
    
        with Pool(processes=best_workers) as pool:
            # warm-up
            pool.map(_worker, chunks)
    
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
    
        t_par = statistics.median(times)
    
        # LIF calculation
        p = best_workers
        lif = p * (t_par / t_serial) - 1
    
        print(f"{n_chunks:6d}    {t_par:.3f}    {lif:.3f}")
    



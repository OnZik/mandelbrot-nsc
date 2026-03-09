# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:18 2026

@author: zikan
"""


from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

@njit
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

@njit
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
    max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks) # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)
    
    return np.vstack(parts)

if __name__ == '__main__':
    result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, n_workers=4)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25], cmap='inferno', origin='lower', aspect='equal')
    
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    



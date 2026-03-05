# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:18 2026

@author: zikan
"""


from numba import njit
import numpy as np
import matplotlib.pyplot as plt

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


plt.imshow(mandelbrot_serial(1024,-2, 1, -1.5, 1.5), cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
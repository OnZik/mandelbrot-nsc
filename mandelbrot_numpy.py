# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:36:54 2026

@author: zikan
"""



"""
Mandelbrot Set Generator
Author : Ondrej Zikan
Course : Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time

max_iter = 100
  
def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height):
    
    x = np.linspace ( xmin , xmax, width) # 1024 x- values
    y = np.linspace ( ymin , ymax , height) # 1024 y- values
    X , Y = np.meshgrid (x , y) # 2D grids
    C = X + 1j* Y # Complex grid
    
    rows, cols = C.shape
    
    M = np.zeros(C.shape, dtype=int)
    
    Z = np.zeros(C.shape, dtype=complex)    
    
    iter_count = 0
    
    while (iter_count < max_iter):
        
        mask = np.abs(Z) <= 2

        Z[mask] = Z[mask]**2 + C[mask]
        
        iter_count+=1
    
        M[mask] += 1
    
    return M
            

start = time.time ()
result = compute_mandelbrot( -2 , 1, -1.5 , 1.5 , 1024 , 1024)
elapsed = time.time() - start
print (f" Computation took { elapsed :.3f} seconds ")

plt.imshow(result, cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
plt.savefig("mandelbrot.png")



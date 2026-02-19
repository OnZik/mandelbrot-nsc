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

def mandelbrod_point(c):
    z = 0
    n_iter = 0
    for n in range(max_iter):
        n_iter = n
        z = z*z + c
        if (abs(z) > 2):
            break;
            
    return n_iter
  
def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height):
    
    x = np.linspace ( xmin , xmax, width) # 1024 x- values
    y = np.linspace ( ymin , ymax , height) # 1024 y- values
    X , Y = np.meshgrid (x , y) # 2D grids
    C = X + 1j* Y # Complex grid
    
    print (f" Shape : {C. shape }") # (1024 , 1024)
    print (f" Type : {C. dtype }") # complex128
    
    rows, cols = C.shape
    

    # Initialize with zeros or NaNs (NaNs are great for spotting unwritten data)
    results_grid = np.full((width, height), np.nan)

    for i in range(rows):
        for j in range(cols):
           results_grid[i, j] = mandelbrod_point(X[i,j] + 1j * Y[i,j])

     
    return results_grid
            
       
start = time.time ()
result = compute_mandelbrot( -2 , 1, -1.5 , 1.5 , 1024 , 1024)
elapsed = time.time() - start
print (f" Computation took { elapsed :.3f} seconds ")

plt.imshow(result, cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
plt.savefig("mandelbrot.png")



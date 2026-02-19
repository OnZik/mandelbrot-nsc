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
#from benchmark import benchmark
import timeit


  
def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    
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

def show_mandelbrot(n):
    M = compute_mandelbrot(-2, 1, -1.5 , 1.5 , n , n , 100)
    
    plt.imshow(M, cmap='viridis')
    plt.colorbar()
    plt.title("Mandelbrot")
    plt.show
    # plt.savefig("mandelbrot.png")
    

# t , M = benchmark (compute_mandelbrot , -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)
       
def show_comp_time():
    grid_sizes = [256, 512, 1024, 2048, 4096]
    runtimes = []  
    for n in grid_sizes:
        t = timeit.timeit(lambda: compute_mandelbrot(-2, 1, -1.5 , 1.5 , n , n , 100), number=1)
        runtimes.append(t)
        print(f"Grid: {n}x{n}, Time: {t:.4f} seconds")
    
    # plot grid size vs. time
    plt.figure()
    plt.plot(grid_sizes, runtimes)
    plt.xlabel("Grid Size")
    plt.ylabel("Runtime")
    plt.show()
    
show_comp_time()
show_mandelbrot(1024)
        
        










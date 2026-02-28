# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 12:37:46 2026

@author: zikan
"""

import numpy as np, time
import matplotlib.pyplot as plt
#from benchmark import benchmark
from numba import njit, int32, complex128

@njit
def mandelbrot_point(c: complex128 , max_iter: int32) -> int32:
    z = 0j
    n_iter = 0
    for n in range(max_iter):
        n_iter = n
        z = z*z + c
        if z.real * z.real + z.imag * z.imag > 4.0:
            break;
    return n_iter

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    x_points = np.linspace(xmin, xmax, width)
    y_points = np.linspace(ymin, ymax, height)
    
    output = np.zeros ((height, width), dtype = np.int32)
    
    for x, real in enumerate(y_points):
       
        for y, imag in enumerate(x_points):
            iter_count = mandelbrot_point(real + imag*1j, max_iter)
           
            output[x][y] = iter_count
     
    return output

@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    x_points = np.linspace(xmin, xmax, width)
    y_points = np.linspace(ymin, ymax, height)
    
    output = np.zeros ((height, width), dtype = np.int32)

    for x, real in enumerate(y_points):      
        for y, imag in enumerate(x_points):
            
            c = real + imag*1j
            z = 0j
            n_iter = 0
            for n in range(max_iter):
                n_iter = n
                z = z*z + c
                if z.real *z.real + z.imag *z.imag > 4.0:
                    break;
           
            output[x][y] = n_iter
     
    return output

@njit
def mandelbrot_numba_typed ( xmin , xmax , ymin , ymax , width , height , max_iter = 100 , dtype = np.float64 ):
    x_points = np.linspace ( xmin , xmax , width ). astype ( dtype )
    y_points = np.linspace ( ymin , ymax , height ) . astype ( dtype )
    output = np.zeros (( height , width ) , dtype = np.int32 )
    for x, real in enumerate(y_points):   
        for y, imag in enumerate(x_points):
            iter_count = mandelbrot_point(real + imag*1j, max_iter)        
            output[x][y] = iter_count    
    return output
        
for dtype in [np.float32 , np.float64 ]:
    t0 = time.perf_counter()
    mandelbrot_numba_typed ( -2 , 1, -1.5 , 1.5 , 1024 , 1024 , dtype = dtype )
    print (f"{ dtype.__name__ }: { time.perf_counter()-t0:.3f}s")



r32 = mandelbrot_numba_typed ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 , dtype = np.float32 )
r64 = mandelbrot_numba_typed ( -2 , 1 , -1.5 , 1.5 , 1024 , 1024 , dtype = np.float64 )
fig , axes = plt.subplots (1 , 2, figsize =(12 , 4) )
for ax , result , title in zip (axes, [r32 ,r64], ['float32', 'float64 (ref)']):
    ax.imshow(result , cmap ='hot')
    ax.set_title(title); ax.axis ('off')
plt.savefig ('precision_comparison.png', dpi =150)

print (f" Max diff float32 vs float64 : {np.abs(r32-r64 ).max ()}")
#_ = mandelbrot_naive_numba( -2 , 1, -1.5 , 1.5 , 64 , 64, 100)
#_ = mandelbrot_hybrid( -2 , 1, -1.5 , 1.5 , 64 , 64, 100)

#t_full , M = benchmark (mandelbrot_naive_numba , -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)
#t_hybrid , M = benchmark (mandelbrot_hybrid , -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)


#print (f" Hybrid : { t_hybrid :.3f}s")
#print (f" Fully compiled : { t_full :.3f}s")
#print (f" Ratio : { t_hybrid / t_full :.1f}x")


#plt.imshow(M, cmap='viridis')
#plt.colorbar()
#plt.title("Mandelbrot")
#plt.show
#plt.savefig("mandelbrot.png")



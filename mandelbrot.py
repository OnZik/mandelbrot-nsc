

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
    
    x_points = np.linspace(xmin, xmax, width)
    y_points = np.linspace(ymin, ymax, height)
    
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for x, real in enumerate(y_points):
       
        for y, imag in enumerate(x_points):
            iter_count = mandelbrod_point(real + imag*1j)
           
            output[x][y] = iter_count
     
    return output
            
       
start = time.time ()
result = compute_mandelbrot( -2 , 1, -1.5 , 1.5 , 1024 , 1024)
elapsed = time.time() - start
print (f" Computation took { elapsed :.3f} seconds ")

plt.imshow(result, cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
plt.savefig("mandelbrot.png")



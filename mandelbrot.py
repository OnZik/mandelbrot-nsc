

"""
Mandelbrot Set Generator
Author : Ondrej Zikan
Course : Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from benchmark import benchmark

def mandelbrod_point(c, max_iter):
    z = 0
    n_iter = 0
    for n in range(max_iter):
        n_iter = n
        z = z*z + c
        if (abs(z) > 2):
            break;
            
    return n_iter
  
def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    x_points = np.linspace(xmin, xmax, width)
    y_points = np.linspace(ymin, ymax, height)
    
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for x, real in enumerate(y_points):
       
        for y, imag in enumerate(x_points):
            iter_count = mandelbrod_point(real + imag*1j, max_iter)
           
            output[x][y] = iter_count
     
    return output

t , M = benchmark (compute_mandelbrot , -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)

plt.imshow(M, cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
plt.savefig("mandelbrot.png")



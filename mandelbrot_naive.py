

"""
Mandelbrot Set Generator
Author : Ondrej Zikan
Course : Numerical Scientific Computing 2026
"""

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point_naive(c, max_iter):
    z = 0
    n_iter = 0
    for n in range(max_iter):
        n_iter = n
        z = z*z + c
        if (abs(z) > 2):
            break;
            
    return n_iter + 1

def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    x_points = np.linspace(xmin, xmax, width)
    y_points = np.linspace(ymin, ymax, height)
    
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for x, imag in enumerate(y_points):
       
        for y, real in enumerate(x_points):
            iter_count = mandelbrot_point_naive(real + imag*1j, max_iter)
           
            output[x][y] = iter_count
     
    return output

#t , output = benchmark (mandelbrot_naive , -2, 1, -1.5 , 1.5 , 512 , 512 , 100)

#if __name__ == "__main__":    
#    mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512, 100)

output =  mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024, 100)
plt.imshow(output, cmap='viridis')
plt.colorbar()
plt.title("Mandelbrot")
plt.show
#plt.savefig("mandelbrot.png")



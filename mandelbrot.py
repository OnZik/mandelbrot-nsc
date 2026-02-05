
import numpy as np
"""
Mandelbrot Set Generator
Author : Ondrej Zikan
Course : Numerical Scientific Computing 2026
"""
max_iter = 100

width, height = 100, 100

# Define the range of x and y
xmin, xmax = -2, 1
ymin, ymax = -1.5, 1.5

# Create evenly spaced values for x and y
x_points = np.linspace(xmin, xmax, width)
y_points = np.linspace(ymin, ymax, height)

def mandelbrod_point(c):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if (abs(z) > 2):
            return n
    return n
  
def compute_mandelbrot():
    array = [] 
    for x in x_points:
        for y in y_points:
            z = x + y*1j
            iter_count = mandelbrod_point(z)
            array.append(iter_count)
            
    return array
            
       
print(compute_mandelbrot());

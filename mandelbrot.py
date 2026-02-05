"""
Mandelbrot Set Generator
Author : Ondrej Zikan
Course : Numerical Scientific Computing 2026
"""
max_iter = 100

def mandelbrod_point(c):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if (abs(z) > 2):
            return n
    return n
  
        
print(mandelbrod_point(1 + 1j));
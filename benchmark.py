# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:00:55 2026

@author: zikan
"""
import time, statistics
from mandelbrot_numba import mandelbrot_naive_numba
from mandelbrot_numpy import mandelbrot_numpy
from mandelbrot_naive import mandelbrot_naive

def benchmark ( func , *args , n_runs = 5) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time.perf_counter ()
        func (* args )
        times.append ( time.perf_counter() - t0 )
    
    median_t = statistics.median ( times )
    
    #print (f" Median : { median_t :.4f}s "
    #      f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    
    return median_t


args = ( -2 , 1, -1.5 , 1.5 , 1024 , 1024, 100)
t_naive = benchmark( mandelbrot_naive, *args)
t_numpy = benchmark( mandelbrot_numpy, *args)
t_numba = benchmark( mandelbrot_naive_numba, *args)
print (f" Naive: {t_naive:.3f}s")
print (f" NumPy: {t_numpy:.3f}s ({t_naive/t_numpy:.1f}x)")
print (f" Numba: {t_numba:.3f}s ({t_naive/t_numba:.1f}x)")
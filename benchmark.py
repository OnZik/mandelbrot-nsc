# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:00:55 2026

@author: zikan
"""
import time, statistics

def benchmark ( func , * args , n_runs =3) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time . perf_counter ()
        result = func (* args )
        times . append ( time . perf_counter () - t0 )
    
    median_t = statistics . median ( times )
    
    print (f" Median : { median_t :.4f}s "
           f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    
    return median_t , result
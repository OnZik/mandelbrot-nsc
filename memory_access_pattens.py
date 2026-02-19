# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:54:02 2026

@author: zikan
"""

import numpy as np
from benchmark import benchmark

N = 10000

A = np.random.rand(N, N) 
A_f = np.asfortranarray(A)

def row_sums():
    for i in range(N): 
        s = np.sum(A_f[i, :])

def col_sums():
    for j in range(N): 
        s = np.sum(A_f[:, j])

        

t , M = benchmark (row_sums)


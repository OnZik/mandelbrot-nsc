# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:27:33 2026

@author: zikan
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration
N, MAX_ITER, TAU = 512, 1000, 0.01
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

# Create complex grids for both precisions
C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

# Initialize state
z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)
diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
active = np.ones((N, N), dtype=bool)

# Iteration Loop
for k in range(MAX_ITER):
    if not active.any():
        break
    
    # Update only active pixels
    z32[active] = z32[active]**2 + C32[active]
    z64[active] = z64[active]**2 + C64[active]
    
    # Calculate the Manhattan distance between 32-bit and 64-bit trajectories
    diff = (np.abs(z32.real.astype(np.float64) - z64.real) + 
            np.abs(z32.imag.astype(np.float64) - z64.imag))
    
    # Identify pixels where the trajectories have drifted apart beyond TAU
    newly_diverged = active & (diff > TAU)
    diverge[newly_diverged] = k
    active[newly_diverged] = False

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(diverge, cmap='plasma', origin='lower',
           extent=[-0.7530, -0.7490, 0.0990, 0.1030])
plt.colorbar(label='Iteration where 32-bit and 64-bit drift apart')
plt.title(f'Numerical Trajectory Divergence (τ={TAU})')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
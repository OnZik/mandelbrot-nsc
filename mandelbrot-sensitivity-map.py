# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:30:09 2026

@author: zikan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Configuration
N, MAX_ITER = 512, 1000
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

# Generate complex grid
C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)

# Define perturbation based on 32-bit machine epsilon
eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)

def escape_count(C, max_iter):
    """Computes the iteration count until escape for a complex array."""
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    
    for k in range(max_iter):
        active = ~esc
        if not np.any(active):
            break
        z[active] = z[active]**2 + C[active]
        newly_escaped = active & (np.abs(z) > 2.0)
        cnt[newly_escaped] = k
        esc[newly_escaped] = True
    return cnt

# Calculate base counts and perturbed counts
n_base = escape_count(C, MAX_ITER).astype(float)
n_perturb = escape_count(C + delta, MAX_ITER).astype(float)

# Compute Numerical Condition Number (kappa)
dn = np.abs(n_base - n_perturb)
kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

# Visualization
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')  # Dark gray for non-escaping points
vmax = np.nanpercentile(kappa, 99)

plt.figure(figsize=(10, 8))
img = plt.imshow(kappa, cmap=cmap_k, origin='lower',
                 extent=[-0.7530, -0.7490, 0.0990, 0.1030],
                 norm=LogNorm(vmin=1, vmax=vmax))

plt.colorbar(img, label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
plt.title(r'Condition number approx $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\, n(c))$')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
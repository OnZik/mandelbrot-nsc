import dask
import numpy as np
import time
import statistics
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit

@njit (cache=True)
def mandelbrot_pixel(c_real, c_imag , max_iter):
    z_real = z_imag = 0.0
    for n in range(max_iter):
        if z_real * z_real + z_imag * z_imag > 4.0:
            return n;
        new_z_imag = 2.0*z_real*z_imag + c_imag
        new_z_real = z_real*z_real - z_imag*z_imag + c_real
        
        z_real = new_z_real
        z_imag = new_z_imag
    return max_iter

@njit (cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

# mandelbrot_chunk: assumed to be your @njit(cache=True) function from previous lessons
def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        # Wrap the chunk processing in dask.delayed
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end
        
    # Execute the graph and gather results
    parts = dask.compute(*tasks)
    return np.vstack(parts)

if __name__ == '__main__':
    # Configuration
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    # Setup Dask Cluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    
    # Warm up all workers to ensure JIT compilation is triggered before timing
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    # ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        # print(np.array_equal(ref, result))
        times.append(time.perf_counter() - t0)
        
    
    # Output results
    median_time = statistics.median(times)
    print(f"Dask local (n_chunks=32): {median_time:.3f}s")
    
    # Clean up
    client.close()
    cluster.close()
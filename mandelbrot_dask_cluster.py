import dask
import numpy as np
import time
import statistics
from dask import delayed
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
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
    N, max_iter = 8192, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    # Setup Dask Cluster
    client = Client("tcp://10.92.1.225:8786")
    
    # Warm up all workers to ensure JIT compilation is triggered before timing
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    # ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    
    # # ----------------------------
    # # Serial baseline
    # # ----------------------------
    # t0 = time.perf_counter()
    # mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    # T1 = time.perf_counter() - t0
    # print(f"\nSerial baseline T1 = {T1:.3f}s\n")
    
    # chunk_numbers = [1,2,4,8,16,32,64,128,256,512]
    # results = []
    
    # print(f"{'n_chunks':>10} | {'time (s)':>10} | {'vs 1x':>8} | {'speedup':>8} | {'LIF':>8}")
    # print("-" * 60)
    
    
    # times_to_plot = []
    # chunk_sizes = []
    
    # for n_chunks in chunk_numbers:
    #     times = []
    #     for _ in range(3):
    #         t0 = time.perf_counter()
    #         result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
    #         # print(np.array_equal(ref, result))
    #         times.append(time.perf_counter() - t0)
        
    
    #     # Output results
    #     Tp = statistics.median(times)
    #     n_workers = 12
    #     speedup = T1 / Tp
    #     vs1x = Tp / T1
    #     LIF = n_workers * Tp / T1 - 1

    #     results.append((n_chunks, Tp, LIF))

    #     print(f"{n_chunks:10d} | {Tp:10.3f} | {vs1x:8.3f} | {speedup:8.3f} | {LIF:8.3f}")
        
    # # ----------------------------
    # # Find optimal
    # # ----------------------------
    # best = min(results, key=lambda x: x[1])  # min time
    # best_lif = min(results, key=lambda x: x[2])

    # print("\n--- Optimal ---")
    # print(f"n_chunks (min time): {best[0]}, t_min = {best[1]:.3f}s")
    # print(f"n_chunks (min LIF):  {best_lif[0]}, LIF_min = {best_lif[2]:.3f}")

    # # ----------------------------
    # # Plot
    # # ----------------------------
    # x = []
    # y = []
    # for r in results:
    #     x.append(r[0])
    #     y.append(r[1])


    # plt.figure()
    # plt.plot(x, y, marker='o')
    # plt.xscale("log")
    # plt.xlabel("n_chunks (log scale)")
    # plt.ylabel("Wall time (s)")
    # plt.title("Dask Chunk Sweep")
    # plt.grid()

    
    # Clean up
    client.close()
    
   
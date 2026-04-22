# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:07:58 2026

@author: zikan
"""

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt


KERNEL_SRC_32 = """

__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch
    
    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;
    
    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

KERNEL_SRC_64 = """

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch
    
    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;
    
    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

N, MAX_ITER = 4096, 100

X_MIN, X_MAX = 0.2, 0.5
Y_MIN, Y_MAX = 0.2, 0.5

RUNS = 3

def benchmark(func, *args, runs=3):
    """Return median wall time (seconds) over `runs` calls."""
    import statistics
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def mandelbrot_gpu_f32(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue):
       
    prog  = cl.Program(ctx, KERNEL_SRC_32).build()
    
    image_f32 = np.zeros((N, N), dtype=np.int32)
    image_f32_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_f32.nbytes)
    
    cl.enqueue_copy(queue, image_f32, image_f32_dev)
    queue.finish()
    
    # --- Warm up (first launch triggers a kernel compile) ---
    prog.mandelbrot_f32(queue, (N, N), None, image_f32_dev,
                    np.float32(X_MIN), np.float32(X_MAX),
                    np.float32(Y_MIN), np.float32(Y_MAX),
                    np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    
    return image_f32
 
    
def mandelbrot_gpu_f64(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue):

    prog  = cl.Program(ctx, KERNEL_SRC_64).build()
    
    image_f64 = np.zeros((N, N), dtype=np.int32)
    image_f64_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_f64.nbytes)
    
    cl.enqueue_copy(queue, image_f64, image_f64_dev)
    queue.finish()
    
    # --- Warm up (first launch triggers a kernel compile) ---
    prog.mandelbrot_f64(queue, (N, N), None, image_f64_dev,
                    np.float64(X_MIN), np.float64(X_MAX),
                    np.float64(Y_MIN), np.float64(Y_MAX),
                    np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    
    return image_f64
    # plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches='tight')
    
ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
dev   = ctx.devices[0]
print(f"Device: {dev.name}\n")

# Warm-up pass (triggers kernel compilation)
_ = mandelbrot_gpu_f32(64, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue)

print(f"Benchmarking N={N}, max_iter={MAX_ITER}, {RUNS} runs each:\n")

# --- float32 ---
t32 = benchmark(mandelbrot_gpu_f32, N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue, runs=RUNS)
img_f32 = mandelbrot_gpu_f32(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue)
print(f"  float32: {t32*1e3:.1f} ms")

# --- float64 ---
t64 = None
img_f64 = mandelbrot_gpu_f64(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue)
if img_f64 is not None:
    t64 = benchmark(mandelbrot_gpu_f64, N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue, runs=RUNS)
    print(f"  float64: {t64*1e3:.1f} ms")
    print(f"  Ratio float64/float32: {t64/t32:.2f}x")
    diff = np.abs(img_f32.astype(int) - img_f64.astype(int))
    print(f"  Max pixel difference (f32 vs f64): {diff.max()}")
    
# --- Visualise ---
fig, axes = plt.subplots(1, 2 if img_f64 is not None else 1,
                         figsize=(12 if img_f64 is not None else 6, 5))
if img_f64 is None:
    axes = [axes]

axes[0].imshow(img_f32, cmap='hot', origin='lower')
axes[0].set_title(f"float32  ({t32*1e3:.1f} ms)")
axes[0].axis('off')

if img_f64 is not None:
    axes[1].imshow(img_f64, cmap='hot', origin='lower')
    axes[1].set_title(f"float64  ({t64*1e3:.1f} ms)")
    axes[1].axis('off')
    
plt.suptitle(f"GPU Mandelbrot  N={N}  device: {dev.name}", fontsize=10)
plt.tight_layout()
plt.show()

# mandelbrot_gpu_f32(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue)
mandelbrot_gpu_f32(N, MAX_ITER, X_MIN, X_MAX, Y_MIN, Y_MAX, ctx, queue)
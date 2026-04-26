# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:56:07 2026

@author: zikan
"""

from mandelbrot_numba import mandelbrot_point, mandelbrot_hybrid, mandelbrot_naive_numba
from mandelbrot_naive import (mandelbrot_point_naive, mandelbrot_naive)
from mandelbrot_numpy import mandelbrot_numpy
from mandelbrot_parallel import (mandelbrot_parallel_p, mandelbrot_chunk, mandelbrot_serial)
from mandelbrot_dask import (mandelbrot_dask, mandelbrot_serial_dask)


import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

KNOWN_CASES = [
    (0+0j,    100, 100),   # origin: never escapes
    (5.0+0j,  100,   1),   # far outside, escapes on iteration 1
    (-2.5+0j, 100,   1),   # left tip of set
]
IMPLEMENTATIONS = [mandelbrot_point, mandelbrot_point_naive]

@pytest.mark.parametrize("impl", IMPLEMENTATIONS)          # independent axis
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)  # bundled
def test_iterations_nnumbers(impl, c, max_iter, expected):
    # TESTING IF THE FUNCTION RETURNS NUMBER OF ITERATIONS
    
    assert impl(c, max_iter) == expected


# ----------------------------------------------------
# TEST GRID
KNOWN_CASES_2 = [
    (-2.0, 1.0, -1.25, 1.25, 64, 64, 100),  # origin: never escapes
    (-2.0, 1.0, -1.25, 1.25, 32, 32, 100) 
]
IMPLEMENTATIONS_2 = [mandelbrot_hybrid, mandelbrot_parallel_p, mandelbrot_numpy]
@pytest.mark.parametrize("impl_2", IMPLEMENTATIONS_2) 
@pytest.mark.parametrize("xmin, xmax, ymin, ymax, width, height, max_iter", KNOWN_CASES_2)  # bundled         # independent axi
def test_output_grid(impl_2, xmin, xmax, ymin, ymax, width, height, max_iter):
    expected = mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter)
    
    actual = impl_2(xmin, xmax, ymin, ymax, width, height, max_iter)
    
    np.testing.assert_allclose(actual, expected, atol=1e-7)
  
# -------------------------------------------------
# MULTIPROCESSING
def test_chunk_shape():
    out = mandelbrot_chunk(0, 4, 4, -2, 1, -1, 1, 20)
    assert out.shape == (4, 4)
    
def test_parallel_matches_serial_small():
    N = 16

    serial = mandelbrot_serial(N, -2, 1, -1, 1, 50)

    parallel = mandelbrot_parallel_p(
        -2, 1, -1, 1,
        N, N,
        50,
        n_workers=2,
        n_chunks=4
    )

    assert np.array_equal(serial, parallel)
    
def test_chunk_matches_serial():
    chunk = mandelbrot_chunk(0, 4, 4, -2, 1, -1, 1, 50)
    serial = mandelbrot_serial(4, -2, 1, -1, 1, 50)

    assert np.array_equal(chunk, serial)


# --------------------------------------------------
# DASK
def test_dask_matches_serial():
    N = 16

    serial = mandelbrot_serial_dask(N, -2, 1, -1, 1, 50)
    dask_result = mandelbrot_dask(N, -2, 1, -1, 1, 50, n_chunks=4)

    assert np.array_equal(serial, dask_result)
    
def test_dask_future_submit():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)

    try:
        future = client.submit(mandelbrot_chunk, 0, 4, 4, -2, 1, -1, 1, 50)
        result = client.gather(future)

        expected = mandelbrot_chunk(0, 4, 4, -2, 1, -1, 1, 50)

        assert np.array_equal(result, expected)

    finally:
        client.close()
        cluster.close()

    
    
    
    
    
    
    
    
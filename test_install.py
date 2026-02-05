## AI script ##

# check_versions.py

# 1. Scientific computing
import numpy
import matplotlib
import scipy

# 2. JIT compilation
import numba

# 3. Testing
import pytest
import pytest_cov  # note: pytest-cov is imported as pytest_cov

# 4. Parallel computing
import dask
import distributed

def print_versions():
    print("=== Scientific Computing ===")
    print(f"numpy: {numpy.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"scipy: {scipy.__version__}\n")

    print("=== JIT Compilation ===")
    print(f"numba: {numba.__version__}\n")

    print("=== Testing ===")
    print(f"pytest: {pytest.__version__}")
    print(f"pytest-cov: {pytest_cov.__version__}\n")

    print("=== Parallel Computing ===")
    print(f"dask: {dask.__version__}")
    print(f"distributed: {distributed.__version__}\n")

if __name__ == "__main__":
    print_versions()
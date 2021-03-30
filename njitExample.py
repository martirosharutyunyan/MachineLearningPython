from time import perf_counter
from numba import jit, njit, prange, float64
from math import sqrt
from numpy import array

@njit(fastmath=True,cache=True,parallel=True)
def isPrime(num):
    if num == 2:
        return True
    if num <= 1 or not num % 2:
        return False
    for i in prange(3,int(sqrt(num))+1,2):
        if not num % i:
            return False
    return True

@njit(fastmath=True,cache=True,parallel=True)
def run_program(N):
    for i in prange(N):
        isPrime(i)

    
start = perf_counter()
N = 10000000
run_program(N)
print(perf_counter()-start)



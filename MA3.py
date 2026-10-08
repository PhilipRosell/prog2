""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:
"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from time import sleep as pause
from numba import njit
import multiprocessing as mp


# Exc1
def approximate_pi(n):
    # n is the number of points
    n_cor = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)]
    ns_cor = [(x, y) for x, y in n_cor if m.sqrt(x**2 + y**2) > 1]
    nc_cor = [(x, y) for x, y in n_cor if m.sqrt(x**2 + y**2) <= 1]

    # create a figure
    fig, ax = plt.subplots()

    nc_xcor, nc_ycor = zip(*nc_cor)
    ax.scatter(nc_xcor, nc_ycor , c='red', s=1)

    ns_xcor, ns_ycor = zip(*ns_cor)
    ax.scatter(ns_xcor, ns_ycor, c='blue', s=1)
    ax.set_title(f"Approximation of pi for (n={n})")

    return 4 * len(nc_cor) / n, fig

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points

    n_cor = [[random.uniform(-1, 1) for j in range(d)] for i in range(n)]
    sphere_cor = list(filter(lambda x: sum(i**2 for i in x) <= 1, n_cor))

    # d is the number of dimensions of the sphere 
    return len(sphere_cor) / n * 2 ** d

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points

    # d is the number of dimensions of the sphere 
    return m.pi ** (d / 2) / m.gamma(d / 2 + 1)

#Exc3: numba version

@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    n_cor = [[random.uniform(-1, 1) for j in range(d)] for i in range(n)]
    sphere_cor = [x for x in n_cor if sum([i**2 for i in x]) <= 1.0]
    return len(sphere_cor) / n * 2 ** d

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    with future.ProcessPoolExecutor() as ex:
        n_per_process = n // np
        n_lst = [n_per_process for i in range(np-1)] + [n % np + n_per_process]
        res = ex.map(sphere_volume, n_lst, [d for i in range(np)])
    return sum(res)

def main():
    # Exc1
    dots = [1000, 10000, 100000]
    approx = []
    for n in dots:
        pi_val, fig = approximate_pi(n)
        approx.append(pi_val)
        plt.savefig(f'pi_approximation_n_{n}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    print("Exc1:")
    print(f"pi ≈ {approx}")


    # Exc2
    n = 100000
    d = 2
    print("\nExc2:")
    print(f"Approximate volume of {d} dimentional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    n = 100000
    d = 11
    print(f"Approximate volume of {d} dimentional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    print("\nExc3:")

    print("\nConventional Python version:")
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Run {i+1}: Time of {d} and {n}: {stop-start}")

    print("\nNumba version:")
    for i in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Run {i+1}: Time of {d} and {n}: {stop-start}")


    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print("\nExc4:")
    print(f"Sequential time of {d} and {n}: {stop-start}")

    start = pc()
    sphere_volume_parallel2(n, d)
    stop = pc()
    print(f"\nParallel time of {d} and {n}: {stop-start}")

    
    

if __name__ == '__main__':
	main()

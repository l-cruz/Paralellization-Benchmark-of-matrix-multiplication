import numpy as np
from scipy import sparse
import time
import os
from multiprocessing import Pool, cpu_count, Lock, Value

sizes = [64, 128, 256, 512, 768, 1024]
runs = 5
sparsity_levels = [0.1, 0.5, 0.9]

#Paralellization
def dense_row(args):
    A, B, i = args
    return np.dot(A[i], B)

def parallel_dense(A, B):
    N = A.shape[0]
    with Pool(cpu_count()) as p:
        rows = p.map(dense_row, [(A, B, i) for i in range(N)])
    return np.vstack(rows)

def sparse_row(args):
    A, B, i = args
    return A[i].dot(B)

def parallel_sparse(A, B):
    N = A.shape[0]
    with Pool(cpu_count()) as p:
        rows = p.map(sparse_row, [(A, B, i) for i in range(N)])
    return sparse.vstack(rows)

def parallel_optimized(A, B):
    return A @ B

lock = Lock()
counter = Value('i', 0)

def increment_counter(_):
    with lock:
        counter.value += 1


if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    with open("results/python_parallel.txt", "w") as f:
        #Dense
        for n in sizes:
            times = []
            f.write(f"\n=== Dense {n}x{n} ===\n")
            print(f"\n=== Dense {n}x{n} ===")

            for r in range(runs):
                A = np.random.rand(n, n)
                B = np.random.rand(n, n)

                start = time.time()
                parallel_dense(A, B)
                end = time.time()

                elapsed = round(end - start, 3)
                times.append(elapsed)

                f.write(f"Run {r+1}: {elapsed} s\n")
                print(f"Run {r+1}: {elapsed} s")

            f.write(f"Mean: {round(sum(times)/runs,3)} s\n")
        #Sparsity
        for sp in sparsity_levels:
            f.write(f"\n=== Sparse (sparsity = {sp}) ===\n")
            print(f"\n=== Sparse (sparsity = {sp}) ===")

            for n in sizes:
                f.write(f"--- N={n} ---\n")
                times = []

                for r in range(runs):
                    density = 1 - sp
                    A = sparse.random(n, n, density=density, format='csr')
                    B = sparse.random(n, n, density=density, format='csr')

                    start = time.time()
                    parallel_sparse(A, B)
                    end = time.time()

                    elapsed = round(end - start, 3)
                    times.append(elapsed)

                    f.write(f"Run {r+1}: {elapsed} s\n")
                    print(f"Run {r+1}: {elapsed} s")

                f.write(f"Mean: {round(sum(times)/runs,3)} s\n")
        #Optimzed
        for n in sizes:
            times = []
            f.write(f"\n=== Optimized {n}x{n} ===\n")
            print(f"\n=== Optimized {n}x{n} ===")

            for r in range(runs):
                A = np.random.rand(n, n)
                B = np.random.rand(n, n)

                start = time.time()
                parallel_optimized(A, B)
                end = time.time()

                elapsed = round(end - start, 3)
                times.append(elapsed)

                f.write(f"Run {r+1}: {elapsed} s\n")
                print(f"Run {r+1}: {elapsed} s")

            f.write(f"Mean: {round(sum(times)/runs,3)} s\n")

    with Pool(cpu_count()) as p:
        p.map(increment_counter, range(10000))

    print("\nSynchronized counter:", counter.value)
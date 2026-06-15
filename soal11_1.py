"""
soal11_1.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.1
Menyelesaikan sistem tridiagonal dengan:
(a) Thomas Algorithm
(b) Gauss-Seidel dengan toleransi es = 5%
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """Menyelesaikan sistem tridiagonal dengan Thomas Algorithm."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).copy()
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float).copy()

    n = len(d)

    # Forward elimination
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def gauss_seidel(A, b, es=5.0, max_iter=100):
    """Menyelesaikan sistem linear dengan metode Gauss-Seidel."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)
    x = np.zeros(n)

    for iterasi in range(1, max_iter + 1):
        x_lama = x.copy()

        for i in range(n):
            jumlah = 0.0

            for j in range(n):
                if j != i:
                    jumlah += A[i, j] * x[j]

            x[i] = (b[i] - jumlah) / A[i, i]

        error = np.zeros(n)

        for i in range(n):
            if x[i] != 0:
                error[i] = abs((x[i] - x_lama[i]) / x[i]) * 100
            else:
                error[i] = np.inf

        if np.all(error < es):
            return x, iterasi

    return x, max_iter


# ==========================================================
# Data Soal 11.1
# ==========================================================

A = np.array([
    [0.8, -0.4, 0.0],
    [-0.4, 0.8, -0.4],
    [0.0, -0.4, 0.8]
])

b = np.array([41.0, 25.0, 105.0])

# Bentuk diagonal untuk Thomas Algorithm
a_diag = np.array([0.0, -0.4, -0.4])    # sub-diagonal
b_diag = np.array([0.8, 0.8, 0.8])      # diagonal utama
c_diag = np.array([-0.4, -0.4, 0.0])    # super-diagonal


# ==========================================================
# Penyelesaian
# ==========================================================

print("=== SOAL 11.1 ===")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)


# Bagian (a): Thomas Algorithm
x_thomas = thomas_algorithm(a_diag, b_diag, c_diag, b)

print("\n(a) Hasil Thomas Algorithm")
print(f"x1 = {x_thomas[0]:.6f}")
print(f"x2 = {x_thomas[1]:.6f}")
print(f"x3 = {x_thomas[2]:.6f}")


# Bagian (b): Gauss-Seidel
x_gs, iterasi = gauss_seidel(A, b, es=5.0)

print("\n(b) Hasil Gauss-Seidel, es = 5%")
print(f"Konvergen pada iterasi ke-{iterasi}")
print(f"x1 = {x_gs[0]:.6f}")
print(f"x2 = {x_gs[1]:.6f}")
print(f"x3 = {x_gs[2]:.6f}")


# Verifikasi dengan numpy
x_exact = np.linalg.solve(A, b)

print("\nVerifikasi numpy.linalg.solve")
print(f"x1 = {x_exact[0]:.6f}")
print(f"x2 = {x_exact[1]:.6f}")
print(f"x3 = {x_exact[2]:.6f}")


# Residual
residual_thomas = np.linalg.norm(np.dot(A, x_thomas) - b, ord=np.inf)
residual_gs = np.linalg.norm(np.dot(A, x_gs) - b, ord=np.inf)

print("\nResidual")
print(f"Thomas Algorithm  = {residual_thomas:.10e}")
print(f"Gauss-Seidel      = {residual_gs:.10e}")
"""
soal11_3.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.3
Menyelesaikan sistem tridiagonal Crank-Nicolson menggunakan Thomas Algorithm.
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """Thomas Algorithm untuk menyelesaikan sistem tridiagonal."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).copy()
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float).copy()

    n = len(d)

    # Forward elimination
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# ==========================================================
# Data Soal 11.3
# ==========================================================

a = np.array([-0.020875, -0.020875, -0.020875])          # sub-diagonal
b = np.array([2.01475, 2.01475, 2.01475, 2.01475])       # diagonal utama
c = np.array([-0.020875, -0.020875, -0.020875])          # super-diagonal
d = np.array([4.175, 0.0, 0.0, 2.0875])                  # ruas kanan

A = np.array([
    [2.01475, -0.020875, 0.0, 0.0],
    [-0.020875, 2.01475, -0.020875, 0.0],
    [0.0, -0.020875, 2.01475, -0.020875],
    [0.0, 0.0, -0.020875, 2.01475]
])


# ==========================================================
# Penyelesaian
# ==========================================================

T = thomas_algorithm(a, b, c, d)
T_exact = np.linalg.solve(A, d)

residual = np.linalg.norm(np.dot(A, T) - d, ord=np.inf)
error = np.max(np.abs(T - T_exact))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.3 ===")

print("\nSolusi dengan Thomas Algorithm:")
print(f"T1 = {T[0]:.6f}")
print(f"T2 = {T[1]:.6f}")
print(f"T3 = {T[2]:.6f}")
print(f"T4 = {T[3]:.6f}")

print("\nVerifikasi dengan numpy.linalg.solve:")
print(f"T1 = {T_exact[0]:.6f}")
print(f"T2 = {T_exact[1]:.6f}")
print(f"T3 = {T_exact[2]:.6f}")
print(f"T4 = {T_exact[3]:.6f}")

print("\nResidual ||A*T - d|| =", residual)
print("Max error Thomas vs numpy =", error)
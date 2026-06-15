"""
soal11_8.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.8
Menyelesaikan sistem tridiagonal dari Soal 11.1 menggunakan
Gauss-Seidel dengan overrelaxation lambda = 1.2 dan es = 5%.
"""

import numpy as np


def gauss_seidel_relaxation(A, b, lam=1.2, es=5.0, max_iter=100):
    """Gauss-Seidel dengan relaxation."""
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

            x_gs = (b[i] - jumlah) / A[i, i]
            x[i] = lam * x_gs + (1 - lam) * x_lama[i]

        error = np.zeros(n)

        for i in range(n):
            if x[i] != 0:
                error[i] = abs((x[i] - x_lama[i]) / x[i]) * 100
            else:
                error[i] = np.inf

        if np.all(error < es):
            return x, iterasi, error

    return x, max_iter, error


# ==========================================================
# Data Soal 11.8: Sistem tridiagonal dari Soal 11.1
# ==========================================================

A = np.array([
    [0.8, -0.4, 0.0],
    [-0.4, 0.8, -0.4],
    [0.0, -0.4, 0.8]
])

b = np.array([41.0, 25.0, 105.0])

lam = 1.2
es = 5.0


# ==========================================================
# Penyelesaian
# ==========================================================

x, iterasi, error = gauss_seidel_relaxation(A, b, lam=lam, es=es)

x_exact = np.linalg.solve(A, b)
residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.8 ===")
print("Metode: Gauss-Seidel dengan overrelaxation")
print(f"lambda = {lam}")
print(f"es     = {es}%")

print(f"\nKonvergen pada iterasi ke-{iterasi}")

print("\nHasil pendekatan:")
print(f"x1 = {x[0]:.6f}")
print(f"x2 = {x[1]:.6f}")
print(f"x3 = {x[2]:.6f}")

print("\nGalat relatif aproksimasi terakhir:")
print(f"ea1 = {error[0]:.6f}%")
print(f"ea2 = {error[1]:.6f}%")
print(f"ea3 = {error[2]:.6f}%")

print("\nSolusi eksak dengan numpy.linalg.solve:")
print(f"x1 = {x_exact[0]:.6f}")
print(f"x2 = {x_exact[1]:.6f}")
print(f"x3 = {x_exact[2]:.6f}")

print("\nResidual ||A*x - b|| =", residual)
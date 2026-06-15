"""
soal11_10.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.10
Mengulang Soal 11.9, tetapi menggunakan metode Jacobi.

Sistem:
15c1 - 3c2 - c3 = 3800
-3c1 + 18c2 - 6c3 = 1200
-4c1 - c2 + 12c3 = 2350
"""

import numpy as np


def jacobi(A, b, es=5.0, max_iter=100):
    """Metode Jacobi dengan kriteria error relatif aproksimasi persen."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)

    for iterasi in range(1, max_iter + 1):
        x_lama = x.copy()
        x_baru = np.zeros(n)

        for i in range(n):
            jumlah = 0.0

            for j in range(n):
                if j != i:
                    jumlah += A[i, j] * x_lama[j]

            x_baru[i] = (b[i] - jumlah) / A[i, i]

        x = x_baru.copy()

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
# Data Soal 11.10
# ==========================================================

A = np.array([
    [15.0, -3.0, -1.0],
    [-3.0, 18.0, -6.0],
    [-4.0, -1.0, 12.0]
])

b = np.array([3800.0, 1200.0, 2350.0])

es = 5.0


# ==========================================================
# Penyelesaian
# ==========================================================

c, iterasi, error = jacobi(A, b, es=es)

c_exact = np.linalg.solve(A, b)
residual = np.linalg.norm(np.dot(A, c) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.10 ===")
print("Metode: Jacobi")
print(f"es = {es}%")

print("\nSistem persamaan:")
print("15c1 - 3c2 - c3 = 3800")
print("-3c1 + 18c2 - 6c3 = 1200")
print("-4c1 - c2 + 12c3 = 2350")

print(f"\nKonvergen pada iterasi ke-{iterasi}")

print("\nHasil konsentrasi reaktor:")
print(f"c1 = {c[0]:.6f} g/m^3")
print(f"c2 = {c[1]:.6f} g/m^3")
print(f"c3 = {c[2]:.6f} g/m^3")

print("\nGalat relatif aproksimasi terakhir:")
print(f"ea1 = {error[0]:.6f}%")
print(f"ea2 = {error[1]:.6f}%")
print(f"ea3 = {error[2]:.6f}%")

print("\nVerifikasi dengan numpy.linalg.solve:")
print(f"c1 = {c_exact[0]:.6f} g/m^3")
print(f"c2 = {c_exact[1]:.6f} g/m^3")
print(f"c3 = {c_exact[2]:.6f} g/m^3")

print("\nResidual ||A*c - b|| =", residual)
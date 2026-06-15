"""
soal11_11.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.11
Menyelesaikan sistem persamaan linear menggunakan metode Gauss-Seidel
hingga percent relative error kurang dari es = 5%.
"""

import numpy as np


def gauss_seidel(A, b, es=5.0, max_iter=100):
    """Metode Gauss-Seidel dengan kriteria error relatif aproksimasi persen."""
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
            return x, iterasi, error

    return x, max_iter, error


# ==========================================================
# Data Soal 11.11
# ==========================================================

A = np.array([
    [10.0, 2.0, -1.0],
    [-3.0, -6.0, 2.0],
    [1.0, 1.0, 5.0]
])

b = np.array([27.0, -61.5, -21.5])

es = 5.0


# ==========================================================
# Penyelesaian
# ==========================================================

x, iterasi, error = gauss_seidel(A, b, es=es)

x_exact = np.linalg.solve(A, b)
residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)

x1, x2, x3 = x
x1_exact, x2_exact, x3_exact = x_exact
e1, e2, e3 = error


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.11 ===")
print("Metode: Gauss-Seidel")
print("es = {}%".format(es))

print("\nSistem persamaan:")
print("10x1 + 2x2 - x3 = 27")
print("-3x1 - 6x2 + 2x3 = -61.5")
print("x1 + x2 + 5x3 = -21.5")

print("\nKonvergen pada iterasi ke-{}".format(iterasi))

print("\nHasil Gauss-Seidel:")
print("x1 = {:.6f}".format(x1))
print("x2 = {:.6f}".format(x2))
print("x3 = {:.6f}".format(x3))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(e1))
print("ea2 = {:.6f}%".format(e2))
print("ea3 = {:.6f}%".format(e3))

print("\nVerifikasi dengan numpy.linalg.solve:")
print("x1 = {:.6f}".format(x1_exact))
print("x2 = {:.6f}".format(x2_exact))
print("x3 = {:.6f}".format(x3_exact))

print("\nResidual ||A*x - b|| =", residual)
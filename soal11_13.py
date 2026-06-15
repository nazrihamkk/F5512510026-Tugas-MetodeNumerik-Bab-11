"""
soal11_13.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.13
Menyelesaikan sistem persamaan linear dengan metode Gauss-Seidel:
(a) tanpa relaksasi
(b) dengan overrelaxation lambda = 1.2

Toleransi: es = 5%
"""

import numpy as np


def gauss_seidel_relaxation(A, b, lam=1.0, es=5.0, max_iter=100):
    """Metode Gauss-Seidel dengan relaxation."""
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
# Data Soal 11.13
# Persamaan asli:
# 2x1 - 6x2 - x3 = -38
# -3x1 - x2 + 7x3 = -34
# -8x1 + x2 - 2x3 = -20
#
# Persamaan disusun ulang agar lebih mudah konvergen:
# -8x1 + x2 - 2x3 = -20
# 2x1 - 6x2 - x3 = -38
# -3x1 - x2 + 7x3 = -34
# ==========================================================

A = np.array([
    [-8.0, 1.0, -2.0],
    [2.0, -6.0, -1.0],
    [-3.0, -1.0, 7.0]
])

b = np.array([-20.0, -38.0, -34.0])

es = 5.0


# ==========================================================
# Penyelesaian
# ==========================================================

x_tanpa, iter_tanpa, error_tanpa = gauss_seidel_relaxation(
    A, b, lam=1.0, es=es
)

x_relax, iter_relax, error_relax = gauss_seidel_relaxation(
    A, b, lam=1.2, es=es
)

x_exact = np.linalg.solve(A, b)

residual_tanpa = np.linalg.norm(np.dot(A, x_tanpa) - b, ord=np.inf)
residual_relax = np.linalg.norm(np.dot(A, x_relax) - b, ord=np.inf)


# ==========================================================
# Simpan nilai agar output tidak error
# ==========================================================

x1_tanpa = x_tanpa[0]
x2_tanpa = x_tanpa[1]
x3_tanpa = x_tanpa[2]

x1_relax = x_relax[0]
x2_relax = x_relax[1]
x3_relax = x_relax[2]

x1_exact = x_exact[0]
x2_exact = x_exact[1]
x3_exact = x_exact[2]

e1_tanpa = error_tanpa[0]
e2_tanpa = error_tanpa[1]
e3_tanpa = error_tanpa[2]

e1_relax = error_relax[0]
e2_relax = error_relax[1]
e3_relax = error_relax[2]


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.13 ===")
print("Metode: Gauss-Seidel")
print("es = {}%".format(es))

print("\nSistem setelah disusun ulang:")
print("-8x1 + x2 - 2x3 = -20")
print("2x1 - 6x2 - x3 = -38")
print("-3x1 - x2 + 7x3 = -34")


print("\n(a) Tanpa relaksasi, lambda = 1.0")
print("Konvergen pada iterasi ke-{}".format(iter_tanpa))
print("x1 = {:.6f}".format(x1_tanpa))
print("x2 = {:.6f}".format(x2_tanpa))
print("x3 = {:.6f}".format(x3_tanpa))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(e1_tanpa))
print("ea2 = {:.6f}%".format(e2_tanpa))
print("ea3 = {:.6f}%".format(e3_tanpa))

print("\nResidual tanpa relaksasi ||A*x - b|| =", residual_tanpa)


print("\n(b) Dengan overrelaxation, lambda = 1.2")
print("Konvergen pada iterasi ke-{}".format(iter_relax))
print("x1 = {:.6f}".format(x1_relax))
print("x2 = {:.6f}".format(x2_relax))
print("x3 = {:.6f}".format(x3_relax))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(e1_relax))
print("ea2 = {:.6f}%".format(e2_relax))
print("ea3 = {:.6f}%".format(e3_relax))

print("\nResidual dengan relaksasi ||A*x - b|| =", residual_relax)


print("\nVerifikasi dengan numpy.linalg.solve:")
print("x1 = {:.6f}".format(x1_exact))
print("x2 = {:.6f}".format(x2_exact))
print("x3 = {:.6f}".format(x3_exact))

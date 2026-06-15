"""
soal11_12.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.12
Menyelesaikan sistem persamaan linear dengan metode Gauss-Seidel:
(a) tanpa relaksasi
(b) dengan relaksasi lambda = 0.95

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
# Data Soal 11.12
# Persamaan sudah disusun ulang agar lebih mudah konvergen
# ==========================================================

A = np.array([
    [6.0, -1.0, -1.0],
    [6.0, 9.0, 1.0],
    [-3.0, 1.0, 12.0]
])

b = np.array([3.0, 40.0, 50.0])

es = 5.0


# ==========================================================
# Penyelesaian
# ==========================================================

x_tanpa, iter_tanpa, error_tanpa = gauss_seidel_relaxation(
    A, b, lam=1.0, es=es
)

x_relax, iter_relax, error_relax = gauss_seidel_relaxation(
    A, b, lam=0.95, es=es
)

x_exact = np.linalg.solve(A, b)

residual_tanpa = np.linalg.norm(np.dot(A, x_tanpa) - b, ord=np.inf)
residual_relax = np.linalg.norm(np.dot(A, x_relax) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.12 ===")
print("Metode: Gauss-Seidel")
print("es = {}%".format(es))

print("\nSistem setelah disusun ulang:")
print("6x1 - x2 - x3 = 3")
print("6x1 + 9x2 + x3 = 40")
print("-3x1 + x2 + 12x3 = 50")


print("\n(a) Tanpa relaksasi, lambda = 1.0")
print("Konvergen pada iterasi ke-{}".format(iter_tanpa))
print("x1 = {:.6f}".format(x_tanpa[0]))
print("x2 = {:.6f}".format(x_tanpa[1]))
print("x3 = {:.6f}".format(x_tanpa[2]))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(error_tanpa[0]))
print("ea2 = {:.6f}%".format(error_tanpa[1]))
print("ea3 = {:.6f}%".format(error_tanpa[2]))

print("\nResidual ||A*x - b|| =", residual_tanpa)


print("\n(b) Dengan relaksasi, lambda = 0.95")
print("Konvergen pada iterasi ke-{}".format(iter_relax))
print("x1 = {:.6f}".format(x_relax[0]))
print("x2 = {:.6f}".format(x_relax[1]))
print("x3 = {:.6f}".format(x_relax[2]))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(error_relax[0]))
print("ea2 = {:.6f}%".format(error_relax[1]))
print("ea3 = {:.6f}%".format(error_relax[2]))

print("\nResidual ||A*x - b|| =", residual_relax)


print("\nVerifikasi dengan numpy.linalg.solve:")
print("x1 = {:.6f}".format(x_exact[0]))
print("x2 = {:.6f}".format(x_exact[1]))
print("x3 = {:.6f}".format(x_exact[2]))
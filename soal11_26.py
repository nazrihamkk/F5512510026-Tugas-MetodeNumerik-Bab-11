"""
soal11_26.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.26
Membuat program user-friendly untuk metode Gauss-Seidel.
Program diuji dengan menduplikasi hasil Example 11.3.

Metode:
Gauss-Seidel Iteration
"""

import numpy as np


def gauss_seidel(A, b, es=5.0, max_iter=100, lam=1.0):
    """
    Menyelesaikan sistem Ax = b dengan metode Gauss-Seidel.

    Parameter:
    A        = matriks koefisien
    b        = vektor ruas kanan
    es       = toleransi galat relatif aproksimasi dalam persen
    max_iter = maksimum iterasi
    lam      = faktor relaksasi
               lam = 1.0 berarti Gauss-Seidel biasa

    Return:
    x        = solusi pendekatan
    iterasi  = jumlah iterasi
    error    = galat relatif aproksimasi terakhir
    """
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
# Data Uji Example 11.3
# ==========================================================
# Sistem:
# 3x1 - 0.1x2 - 0.2x3 = 7.85
# 0.1x1 + 7x2 - 0.3x3 = -19.3
# 0.3x1 - 0.2x2 + 10x3 = 71.4
# ==========================================================

A = np.array([
    [3.0, -0.1, -0.2],
    [0.1, 7.0, -0.3],
    [0.3, -0.2, 10.0]
])

b = np.array([7.85, -19.3, 71.4])

es = 5.0
lam = 1.0


# ==========================================================
# Penyelesaian
# ==========================================================

x, iterasi, error = gauss_seidel(A, b, es=es, lam=lam)

x_exact = np.linalg.solve(A, b)

hasil_cek = np.dot(A, x)
residual = np.linalg.norm(hasil_cek - b, ord=np.inf)

max_error = np.max(np.abs(x - x_exact))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.26 ===")
print("Program Gauss-Seidel Method")

print("\nSistem persamaan:")
print("3x1 - 0.1x2 - 0.2x3 = 7.85")
print("0.1x1 + 7x2 - 0.3x3 = -19.3")
print("0.3x1 - 0.2x2 + 10x3 = 71.4")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)

print("\nParameter:")
print("es = {}%".format(es))
print("lambda = {}".format(lam))

print("\nHasil Gauss-Seidel:")
print("Konvergen pada iterasi ke-{}".format(iterasi))
print("x1 = {:.6f}".format(x[0]))
print("x2 = {:.6f}".format(x[1]))
print("x3 = {:.6f}".format(x[2]))

print("\nGalat relatif aproksimasi terakhir:")
print("ea1 = {:.6f}%".format(error[0]))
print("ea2 = {:.6f}%".format(error[1]))
print("ea3 = {:.6f}%".format(error[2]))

print("\nVerifikasi dengan numpy.linalg.solve:")
print("x1 = {:.6f}".format(x_exact[0]))
print("x2 = {:.6f}".format(x_exact[1]))
print("x3 = {:.6f}".format(x_exact[2]))

print("\nResidual ||A*x - b|| = {:.10e}".format(residual))
print("Max error Gauss-Seidel vs numpy = {:.10e}".format(max_error))

print("\nKesimpulan:")
print("Program Gauss-Seidel berhasil dibuat dan hasilnya sesuai dengan Example 11.3.")
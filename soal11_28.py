"""
soal11_28.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.28
Menyelesaikan sistem pentadiagonal dengan bandwidth 5.
"""

import numpy as np


def pentadiagonal_solver(A, b):
    """Menyelesaikan sistem pentadiagonal tanpa pivoting."""
    A = np.array(A, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    n = len(b)

    # Eliminasi maju hanya pada pita pentadiagonal
    for k in range(n - 1):
        if abs(A[k, k]) < 1e-15:
            raise ZeroDivisionError("Pivot bernilai nol.")

        for i in range(k + 1, min(n, k + 3)):
            if A[i, k] != 0:
                faktor = A[i, k] / A[k, k]
                batas = min(n, k + 3)

                A[i, k:batas] = A[i, k:batas] - faktor * A[k, k:batas]
                b[i] = b[i] - faktor * b[k]

    # Substitusi mundur
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        batas = min(n, i + 3)
        jumlah = np.dot(A[i, i + 1:batas], x[i + 1:batas])
        x[i] = (b[i] - jumlah) / A[i, i]

    return x


# ==========================================================
# Data Soal 11.28
# ==========================================================

A = np.array([
    [8.0, -2.0, -1.0, 0.0, 0.0],
    [-2.0, 9.0, -4.0, -1.0, 0.0],
    [-1.0, -3.0, 7.0, -1.0, -2.0],
    [0.0, -4.0, -2.0, 12.0, -5.0],
    [0.0, 0.0, -7.0, -3.0, 15.0]
])

b = np.array([5.0, 2.0, 0.0, 1.0, 5.0])


# ==========================================================
# Penyelesaian
# ==========================================================

x = pentadiagonal_solver(A, b)
x_numpy = np.linalg.solve(A, b)

hasil_cek = np.dot(A, x)
residual = np.linalg.norm(hasil_cek - b, ord=np.inf)
max_error = np.max(np.abs(x - x_numpy))

x1, x2, x3, x4, x5 = x
xn1, xn2, xn3, xn4, xn5 = x_numpy


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.28 ===")
print("Solver sistem pentadiagonal bandwidth 5")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)

print("\nSolusi dengan pentadiagonal solver:")
print("x1 = {:.6f}".format(x1))
print("x2 = {:.6f}".format(x2))
print("x3 = {:.6f}".format(x3))
print("x4 = {:.6f}".format(x4))
print("x5 = {:.6f}".format(x5))

print("\nVerifikasi dengan numpy.linalg.solve:")
print("x1 = {:.6f}".format(xn1))
print("x2 = {:.6f}".format(xn2))
print("x3 = {:.6f}".format(xn3))
print("x4 = {:.6f}".format(xn4))
print("x5 = {:.6f}".format(xn5))

print("\nVerifikasi A dikali x:")
print(np.round(hasil_cek, 6))

print("\nResidual ||A*x - b|| = {:.10e}".format(residual))
print("Max error solver vs numpy = {:.10e}".format(max_error))

print("\nKesimpulan:")
print("Solver pentadiagonal berhasil menyelesaikan sistem dengan benar.")
"""
soal11_22.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.22
Menuliskan sistem persamaan ke bentuk matriks,
menyelesaikan sistem, menghitung transpose, dan invers matriks koefisien.

Persamaan:
50 = 5x3 - 7x2
4x2 + 7x3 + 30 = 0
x1 - 7x3 = 40 - 3x2 + 5x1
"""

import numpy as np


# ==========================================================
# Bentuk standar Ax = b
# ==========================================================
# 50 = 5x3 - 7x2
# 0x1 - 7x2 + 5x3 = 50
#
# 4x2 + 7x3 + 30 = 0
# 0x1 + 4x2 + 7x3 = -30
#
# x1 - 7x3 = 40 - 3x2 + 5x1
# -4x1 + 3x2 - 7x3 = 40
# ==========================================================

A = np.array([
    [0.0, -7.0, 5.0],
    [0.0, 4.0, 7.0],
    [-4.0, 3.0, -7.0]
], dtype=float)

b = np.array([50.0, -30.0, 40.0], dtype=float)


# ==========================================================
# Perhitungan
# ==========================================================

x = np.linalg.solve(A, b)

A_transpose = np.transpose(A)
A_inverse = np.linalg.inv(A)

hasil_cek = np.dot(A, x)
residual = np.linalg.norm(hasil_cek - b, ord=np.inf)


# ==========================================================
# Ambil nilai agar print tidak error
# ==========================================================

x1 = x[0]
x2 = x[1]
x3 = x[2]


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.22 ===")

print("\nBentuk standar persamaan:")
print("0x1 - 7x2 + 5x3 = 50")
print("0x1 + 4x2 + 7x3 = -30")
print("-4x1 + 3x2 - 7x3 = 40")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)

print("\nSolusi sistem Ax = b:")
print("x1 = {:.6f}".format(x1))
print("x2 = {:.6f}".format(x2))
print("x3 = {:.6f}".format(x3))

print("\nTranspose matriks A:")
print(A_transpose)

print("\nInvers matriks A:")
print(np.round(A_inverse, 6))

print("\nVerifikasi A dikali x:")
print(np.round(hasil_cek, 6))

print("\nResidual ||A*x - b|| = {:.10e}".format(residual))

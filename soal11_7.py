"""
soal11_7.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.7
Menghitung dekomposisi Cholesky untuk matriks diagonal:

A = [[9,  0, 0],
     [0, 25, 0],
     [0,  0, 4]]

Kemudian menjelaskan apakah hasilnya masuk akal berdasarkan persamaan Cholesky.
"""

import numpy as np


def cholesky_decomposition(A):
    """Dekomposisi Cholesky manual: A = L L^T."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            jumlah = 0.0

            for k in range(j):
                jumlah += L[i, k] * L[j, k]

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - jumlah)
            else:
                L[i, j] = (A[i, j] - jumlah) / L[j, j]

    return L


# ==========================================================
# Data Soal 11.7
# ==========================================================

A = np.array([
    [9.0, 0.0, 0.0],
    [0.0, 25.0, 0.0],
    [0.0, 0.0, 4.0]
])


# ==========================================================
# Penyelesaian
# ==========================================================

L = cholesky_decomposition(A)
LT = L.T
A_check = np.dot(L, LT)

error = np.max(np.abs(A - A_check))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.7 ===")

print("\nMatriks A:")
print(A)

print("\nMatriks L hasil Cholesky:")
print(np.round(L, 6))

print("\nMatriks L Transpose:")
print(np.round(LT, 6))

print("\nHasil L dikali L Transpose:")
print(np.round(A_check, 6))

print("\nMax error |A - L*L^T| =", error)

print("\nKesimpulan:")
print("Hasilnya masuk akal.")
print("Karena A adalah matriks diagonal positif, maka L juga matriks diagonal.")
print("Setiap elemen diagonal L adalah akar dari elemen diagonal A:")
print("sqrt(9) = 3, sqrt(25) = 5, sqrt(4) = 2.")

if np.allclose(A, A_check):
    print("Validasi: BENAR, karena L dikali L Transpose menghasilkan A.")
else:
    print("Validasi: SALAH.")
"""
soal11_25.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.25
Membuat program user-friendly untuk Dekomposisi Cholesky.
Program diuji dengan menduplikasi hasil Example 11.2.

Dekomposisi Cholesky:
A = L * L^T
"""

import numpy as np


def cholesky_decomposition(A):
    """Menghitung dekomposisi Cholesky A = L * L^T."""
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matriks harus berbentuk persegi.")

    if not np.allclose(A, A.T):
        raise ValueError("Matriks harus simetris.")

    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            jumlah = 0.0

            for k in range(j):
                jumlah += L[i, k] * L[j, k]

            if i == j:
                nilai = A[i, i] - jumlah

                if nilai <= 0:
                    raise ValueError("Matriks harus positif definit.")

                L[i, j] = np.sqrt(nilai)
            else:
                L[i, j] = (A[i, j] - jumlah) / L[j, j]

    return L


# ==========================================================
# Data Uji Example 11.2
# ==========================================================

A = np.array([
    [6.0, 15.0, 55.0],
    [15.0, 55.0, 225.0],
    [55.0, 225.0, 979.0]
])


# ==========================================================
# Penyelesaian
# ==========================================================

L = cholesky_decomposition(A)
L_transpose = np.transpose(L)
A_check = np.dot(L, L_transpose)
error = np.max(np.abs(A - A_check))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.25 ===")
print("Program Cholesky Decomposition")

print("\nMatriks A:")
print(A)

print("\nMatriks L hasil Cholesky:")
print(np.round(L, 6))

print("\nMatriks L Transpose:")
print(np.round(L_transpose, 6))

print("\nVerifikasi L dikali L Transpose:")
print(np.round(A_check, 6))

print("\nMax error |A - L*L^T| = {:.10e}".format(error))

if np.allclose(A, A_check):
    print("\nKesimpulan: Dekomposisi Cholesky benar, karena L*L^T = A.")
else:
    print("\nKesimpulan: Dekomposisi Cholesky salah.")
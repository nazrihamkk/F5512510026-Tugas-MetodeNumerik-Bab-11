"""
soal11_2.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.2
Menentukan invers matriks Example 11.1 menggunakan LU Decomposition
dan unit vectors, yaitu menyelesaikan A x = e_i untuk setiap kolom identitas.
"""

import numpy as np


def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            jumlah = 0.0
            for k in range(i):
                jumlah += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - jumlah

        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-15:
                raise ZeroDivisionError("Pivot nol, LU tanpa pivoting gagal.")

            jumlah = 0.0
            for k in range(i):
                jumlah += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - jumlah) / U[i, i]

    return L, U


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        jumlah = 0.0
        for j in range(i):
            jumlah += L[i, j] * y[j]
        y[i] = (b[i] - jumlah) / L[i, i]

    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        jumlah = 0.0
        for j in range(i + 1, n):
            jumlah += U[i, j] * x[j]
        x[i] = (y[i] - jumlah) / U[i, i]

    return x


def inverse_lu(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L, U = lu_decomposition(A)
    I = np.eye(n)
    A_inv = np.zeros((n, n))

    for i in range(n):
        y = forward_substitution(L, I[:, i])
        x = backward_substitution(U, y)
        A_inv[:, i] = x

    return A_inv, L, U


# ==========================================================
# Data Soal 11.2: Matriks Example 11.1
# ==========================================================

A = np.array([
    [2.04, -1.00,  0.00,  0.00],
    [-1.00,  2.04, -1.00,  0.00],
    [0.00, -1.00,  2.04, -1.00],
    [0.00,  0.00, -1.00,  2.04]
])


# ==========================================================
# Penyelesaian
# ==========================================================

A_inv, L, U = inverse_lu(A)
identitas = np.dot(A, A_inv)
error = np.max(np.abs(A_inv - np.linalg.inv(A)))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.2 ===")

print("\nMatriks A:")
print(A)

print("\nMatriks L:")
print(np.round(L, 6))

print("\nMatriks U:")
print(np.round(U, 6))

print("\nInvers A menggunakan LU dan unit vectors:")
print(np.round(A_inv, 6))

print("\nVerifikasi A dikali A_inv:")
print(np.round(identitas, 6))

print("\nMax error terhadap numpy.linalg.inv:")
print(error)
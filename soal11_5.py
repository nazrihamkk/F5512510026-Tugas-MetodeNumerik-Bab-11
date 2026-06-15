"""
soal11_5.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.5
Menyelesaikan sistem simetris dengan Dekomposisi Cholesky.

A a = b
A = L L^T
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


def forward_substitution(L, b):
    """Menyelesaikan L y = b."""
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        jumlah = 0.0

        for j in range(i):
            jumlah += L[i, j] * y[j]

        y[i] = (b[i] - jumlah) / L[i, i]

    return y


def backward_substitution(U, y):
    """Menyelesaikan U x = y."""
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        jumlah = 0.0

        for j in range(i + 1, n):
            jumlah += U[i, j] * x[j]

        x[i] = (y[i] - jumlah) / U[i, i]

    return x


# ==========================================================
# Data Soal 11.5
# ==========================================================

A = np.array([
    [6.0, 15.0, 55.0],
    [15.0, 55.0, 225.0],
    [55.0, 225.0, 979.0]
])

b = np.array([152.6, 585.6, 2488.8])


# ==========================================================
# Penyelesaian
# ==========================================================

L = cholesky_decomposition(A)
y = forward_substitution(L, b)
a = backward_substitution(L.T, y)

hasil = np.dot(A, a)
residual = np.linalg.norm(hasil - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.5 ===")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)

print("\nMatriks L hasil Cholesky:")
print(np.round(L, 6))

print("\nVektor y dari L y = b:")
print(np.round(y, 6))

print("\nSolusi akhir:")
print(f"a0 = {a[0]:.6f}")
print(f"a1 = {a[1]:.6f}")
print(f"a2 = {a[2]:.6f}")

print("\nVerifikasi A dikali a:")
print(np.round(hasil, 6))

print("\nResidual ||A*a - b|| =", residual)
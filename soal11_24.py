"""
soal11_24.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.24
Membuat program user-friendly untuk menyelesaikan sistem tridiagonal
menggunakan Thomas Algorithm.

Program diuji dengan sistem tridiagonal dari Example 11.1.
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """
    Menyelesaikan sistem tridiagonal menggunakan Thomas Algorithm.

    Parameter:
    a = sub-diagonal, panjang n-1
    b = diagonal utama, panjang n
    c = super-diagonal, panjang n-1
    d = ruas kanan, panjang n

    Return:
    x = solusi sistem tridiagonal
    """
    a = np.array(a, dtype=float).copy()
    b = np.array(b, dtype=float).copy()
    c = np.array(c, dtype=float).copy()
    d = np.array(d, dtype=float).copy()

    n = len(d)

    if len(b) != n:
        raise ValueError("Panjang diagonal utama b harus sama dengan panjang d.")

    if len(a) != n - 1:
        raise ValueError("Panjang sub-diagonal a harus n-1.")

    if len(c) != n - 1:
        raise ValueError("Panjang super-diagonal c harus n-1.")

    # Forward elimination
    for i in range(1, n):
        if b[i - 1] == 0:
            raise ZeroDivisionError("Terjadi pembagian dengan nol pada proses eliminasi.")

        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back substitution
    x = np.zeros(n)

    if b[n - 1] == 0:
        raise ZeroDivisionError("Terjadi pembagian dengan nol pada substitusi balik.")

    x[n - 1] = d[n - 1] / b[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# ==========================================================
# Data Uji Example 11.1
# ==========================================================
# Sistem:
# 2.04x1 - x2                 = 40.8
# -x1 + 2.04x2 - x3           = 0.8
#      -x2 + 2.04x3 - x4      = 0.8
#             -x3 + 2.04x4    = 200.8
# ==========================================================

a = [-1.0, -1.0, -1.0]
b = [2.04, 2.04, 2.04, 2.04]
c = [-1.0, -1.0, -1.0]
d = [40.8, 0.8, 0.8, 200.8]


# ==========================================================
# Penyelesaian
# ==========================================================

x = thomas_algorithm(a, b, c, d)


# ==========================================================
# Verifikasi dengan numpy
# ==========================================================

A = np.array([
    [2.04, -1.0, 0.0, 0.0],
    [-1.0, 2.04, -1.0, 0.0],
    [0.0, -1.0, 2.04, -1.0],
    [0.0, 0.0, -1.0, 2.04]
])

d_vector = np.array(d, dtype=float)

x_numpy = np.linalg.solve(A, d_vector)

residual = np.linalg.norm(np.dot(A, x) - d_vector, ord=np.inf)
max_error = np.max(np.abs(x - x_numpy))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.24 ===")
print("Program Thomas Algorithm untuk sistem tridiagonal")

print("\nData diagonal:")
print("Sub-diagonal a     =", a)
print("Diagonal utama b   =", b)
print("Super-diagonal c   =", c)
print("Ruas kanan d       =", d)

print("\nSolusi dengan Thomas Algorithm:")
for i in range(len(x)):
    print("x{} = {:.6f}".format(i + 1, x[i]))

print("\nVerifikasi dengan numpy.linalg.solve:")
for i in range(len(x_numpy)):
    print("x{} = {:.6f}".format(i + 1, x_numpy[i]))

print("\nResidual ||A*x - d|| = {:.10e}".format(residual))
print("Max error Thomas vs numpy = {:.10e}".format(max_error))

print("\nKesimpulan:")
print("Program Thomas Algorithm berhasil dibuat dan hasilnya sesuai dengan numpy.")
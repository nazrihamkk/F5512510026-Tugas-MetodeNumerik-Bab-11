"""
soal11_20.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.20
Mengulang Soal 11.19, tetapi menggunakan matriks Vandermonde 6 dimensi.

Diketahui:
x1 = 4, x2 = 2, x3 = 7, x4 = 10, x5 = 3, x6 = 5

Buat matriks Vandermonde 6x6.
Kemudian tentukan spectral condition number dan selesaikan sistem Vc = b,
dengan b adalah jumlah setiap baris V.

Karena b adalah jumlah setiap baris V, maka solusi eksak seharusnya:
c1 = c2 = c3 = c4 = c5 = c6 = 1
"""

import numpy as np


# ==========================================================
# Data Soal 11.20
# ==========================================================

x_data = np.array([4.0, 2.0, 7.0, 10.0, 3.0, 5.0])
n = len(x_data)


# ==========================================================
# Membentuk Matriks Vandermonde
# Baris i: [1, x_i, x_i^2, x_i^3, x_i^4, x_i^5]
# ==========================================================

V = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        V[i, j] = x_data[i] ** j


# ==========================================================
# Membentuk Vektor b
# b adalah jumlah setiap baris V
# ==========================================================

b = np.sum(V, axis=1)


# ==========================================================
# Penyelesaian
# ==========================================================

c = np.linalg.solve(V, b)

c_exact = np.ones(n)

error = np.abs(c - c_exact)

condition_number = np.linalg.cond(V, 2)

digits_lost = np.log10(condition_number)

residual = np.linalg.norm(np.dot(V, c) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.20 ===")

print("\nNilai x:")
print(x_data)

print("\nMatriks Vandermonde V:")
print(np.round(V, 6))

print("\nVektor b:")
print(np.round(b, 6))

print("\nSpectral Condition Number:")
print("{:.6e}".format(condition_number))

print("\nEstimasi digit presisi yang hilang:")
print("{:.2f} digit".format(digits_lost))

print("\nSolusi numerik c:")
for i in range(n):
    print("c{} = {:.10f}".format(i + 1, c[i]))

print("\nError terhadap solusi eksak c = 1:")
for i in range(n):
    print("error c{} = {:.10e}".format(i + 1, error[i]))

print("\nMax error:")
print("{:.10e}".format(np.max(error)))

print("\nResidual ||V*c - b||:")
print("{:.10e}".format(residual))

print("\nAnalisis:")
print("Solusi eksak seharusnya semua c = 1 karena b adalah jumlah setiap baris V.")
print("Matriks Vandermonde memiliki condition number cukup besar.")
print("Artinya matriks cukup ill-conditioned dan beberapa digit presisi dapat hilang.")
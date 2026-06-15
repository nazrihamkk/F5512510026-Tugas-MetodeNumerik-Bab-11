"""
soal11_19.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.19
Menentukan spectral condition number untuk Hilbert matrix 10 dimensi.
Kemudian menyelesaikan sistem Hx = b, dengan b adalah jumlah elemen
pada setiap baris H, sehingga solusi eksak seharusnya x = 1.
"""

import numpy as np


# ==========================================================
# Membentuk Hilbert Matrix 10 x 10
# ==========================================================

n = 10

H = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        H[i, j] = 1.0 / (i + j + 1)


# ==========================================================
# Membentuk vektor b
# b adalah jumlah setiap baris Hilbert matrix
# sehingga solusi eksaknya adalah x = [1, 1, ..., 1]
# ==========================================================

b = np.sum(H, axis=1)


# ==========================================================
# Penyelesaian
# ==========================================================

x = np.linalg.solve(H, b)

x_exact = np.ones(n)

error = np.abs(x - x_exact)

condition_number = np.linalg.cond(H, 2)

digits_lost = np.log10(condition_number)

residual = np.linalg.norm(np.dot(H, x) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.19 ===")

print("\nHilbert Matrix 10 x 10:")
print(np.round(H, 6))

print("\nVektor b:")
print(np.round(b, 10))

print("\nSpectral Condition Number:")
print("{:.6e}".format(condition_number))

print("\nEstimasi digit presisi yang hilang:")
print("{:.2f} digit".format(digits_lost))

print("\nSolusi numerik x:")
for i, nilai in enumerate(x):
    print("x{} = {:.10f}".format(i + 1, nilai))

print("\nError terhadap solusi eksak x = 1:")
for i, nilai_error in enumerate(error):
    print("error x{} = {:.10e}".format(i + 1, nilai_error))

print("\nMax error:")
print("{:.10e}".format(np.max(error)))

print("\nResidual ||H*x - b||:")
print("{:.10e}".format(residual))

print("\nAnalisis:")
print("Hilbert matrix 10 x 10 memiliki condition number yang sangat besar.")
print("Hal ini menunjukkan bahwa matriks sangat ill-conditioned.")
print("Akibatnya, beberapa digit presisi hilang karena round-off error.")
print("Walaupun solusi eksak seharusnya semua x = 1, hasil numerik sedikit menyimpang.")
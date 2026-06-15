"""
soal11_18.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.18
Sebuah perusahaan elektronik memproduksi transistor, resistor, dan computer chips.

Kebutuhan material:
Transistor membutuhkan 4 copper, 1 zinc, 2 glass.
Resistor membutuhkan 3 copper, 3 zinc, 1 glass.
Computer chip membutuhkan 2 copper, 1 zinc, 3 glass.

Persediaan:
Copper = 960
Zinc   = 510
Glass  = 610

Tentukan jumlah transistor, resistor, dan computer chips yang harus diproduksi.
"""

import numpy as np


# ==========================================================
# Model Sistem Persamaan
# ==========================================================
# Misalkan:
# x1 = jumlah transistor
# x2 = jumlah resistor
# x3 = jumlah computer chips
#
# 4x1 + 3x2 + 2x3 = 960   copper
# 1x1 + 3x2 + 1x3 = 510   zinc
# 2x1 + 1x2 + 3x3 = 610   glass
# ==========================================================

A = np.array([
    [4.0, 3.0, 2.0],
    [1.0, 3.0, 1.0],
    [2.0, 1.0, 3.0]
])

b = np.array([960.0, 510.0, 610.0])


# ==========================================================
# Penyelesaian
# ==========================================================

x = np.linalg.solve(A, b)

transistor = x[0]
resistor = x[1]
chip = x[2]

hasil_cek = np.dot(A, x)
residual = np.linalg.norm(hasil_cek - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.18 ===")

print("\nModel sistem persamaan:")
print("4x1 + 3x2 + 2x3 = 960")
print("x1 + 3x2 + x3   = 510")
print("2x1 + x2 + 3x3  = 610")

print("\nKeterangan variabel:")
print("x1 = Transistor")
print("x2 = Resistor")
print("x3 = Computer chips")

print("\nMatriks A:")
print(A)

print("\nVektor b:")
print(b)

print("\nHasil produksi:")
print("Transistor     = {:.0f} unit".format(transistor))
print("Resistor       = {:.0f} unit".format(resistor))
print("Computer chips = {:.0f} unit".format(chip))

print("\nVerifikasi penggunaan material:")
print("Copper = {:.0f}".format(hasil_cek[0]))
print("Zinc   = {:.0f}".format(hasil_cek[1]))
print("Glass  = {:.0f}".format(hasil_cek[2]))

print("\nResidual ||A*x - b|| =", residual)

print("\nKesimpulan:")
print("Jumlah produksi yang harus dibuat adalah:")
print("{:.0f} transistor, {:.0f} resistor, dan {:.0f} computer chips.".format(
    transistor, resistor, chip
))
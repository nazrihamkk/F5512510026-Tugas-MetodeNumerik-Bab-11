"""
soal11_14.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.14
Menggambar ulang Fig. 11.5 untuk kasus slope persamaan 1 dan -1,
lalu menjelaskan hasil penerapan metode Gauss-Seidel.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Sistem Persamaan
# ==========================================================
# Persamaan 1: x1 + x2 = 3  -> x2 = 3 - x1, slope = -1
# Persamaan 2: x1 - x2 = 1  -> x2 = x1 - 1, slope = 1

A = np.array([
    [1.0, 1.0],
    [1.0, -1.0]
])

b = np.array([3.0, 1.0])


# ==========================================================
# Iterasi Gauss-Seidel
# ==========================================================

x = np.array([0.0, 0.0])
jumlah_iterasi = 10

riwayat = [x.copy()]

for iterasi in range(1, jumlah_iterasi + 1):
    x_lama = x.copy()

    # Dari persamaan 1:
    # x1 + x2 = 3  ->  x1 = 3 - x2
    x[0] = 3.0 - x[1]

    # Dari persamaan 2:
    # x1 - x2 = 1  ->  x2 = x1 - 1
    x[1] = x[0] - 1.0

    riwayat.append(x.copy())

riwayat = np.array(riwayat)


# ==========================================================
# Verifikasi Solusi Eksak
# ==========================================================

x_exact = np.linalg.solve(A, b)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.14 ===")

print("\nSistem persamaan:")
print("x1 + x2 = 3   dengan slope = -1")
print("x1 - x2 = 1   dengan slope = 1")

print("\nSolusi eksak:")
print("x1 = {:.6f}".format(x_exact[0]))
print("x2 = {:.6f}".format(x_exact[1]))

print("\nHasil iterasi Gauss-Seidel:")
print("{:<10} {:>12} {:>12}".format("Iterasi", "x1", "x2"))

for i in range(len(riwayat)):
    print("{:<10} {:>12.6f} {:>12.6f}".format(i, riwayat[i, 0], riwayat[i, 1]))

print("\nKesimpulan:")
print("Metode Gauss-Seidel tidak konvergen.")
print("Nilai iterasi bergerak bolak-balik atau berosilasi.")
print("Hal ini terjadi karena sistem tidak memenuhi kondisi konvergensi yang baik.")


# ==========================================================
# Grafik
# ==========================================================

x1_line = np.linspace(-1, 5, 200)

# x2 = 3 - x1
garis_1 = 3 - x1_line

# x2 = x1 - 1
garis_2 = x1_line - 1

plt.figure(figsize=(8, 6))

plt.plot(x1_line, garis_1, label="x1 + x2 = 3, slope = -1")
plt.plot(x1_line, garis_2, label="x1 - x2 = 1, slope = 1")

plt.plot(
    riwayat[:, 0],
    riwayat[:, 1],
    "ro--",
    label="Jalur iterasi Gauss-Seidel"
)

plt.plot(
    x_exact[0],
    x_exact[1],
    "g*",
    markersize=15,
    label="Solusi eksak"
)

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.title("Soal 11.14 - Gauss-Seidel untuk Slope 1 dan -1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()

plt.savefig("soal11_14.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nGrafik berhasil disimpan sebagai: soal11_14.png")
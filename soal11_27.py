"""
soal11_27.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.27
Menyelesaikan persamaan diferensial kanal satu dimensi dengan metode beda hingga.

Persamaan:
0 = D * d2c/dx2 - U * dc/dx - k*c

Diketahui:
D = 2
U = 1
k = 0.2
c(0) = 80
c(10) = 20
dx = 2

Diminta:
Menyelesaikan dari x = 0 sampai x = 10 dan membuat plot c terhadap x.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Data Soal 11.27
# ==========================================================

D = 2.0
U = 1.0
k = 0.2

c_awal = 80.0
c_akhir = 20.0

x_awal = 0.0
x_akhir = 10.0
dx = 2.0


# ==========================================================
# Grid
# ==========================================================

x = np.arange(x_awal, x_akhir + dx, dx)

# Titik total: x = 0, 2, 4, 6, 8, 10
# Titik interior: x = 2, 4, 6, 8
n_interior = len(x) - 2


# ==========================================================
# Koefisien beda hingga
# ==========================================================
# 0 = D*(c(i-1) - 2c(i) + c(i+1))/dx^2
#     - U*(c(i+1) - c(i-1))/(2dx)
#     - k*c(i)
#
# Koefisien:
# c(i-1) = D/dx^2 + U/(2dx)
# c(i)   = -2D/dx^2 - k
# c(i+1) = D/dx^2 - U/(2dx)
# ==========================================================

koef_kiri = D / dx**2 + U / (2 * dx)
koef_tengah = -2 * D / dx**2 - k
koef_kanan = D / dx**2 - U / (2 * dx)


# ==========================================================
# Membentuk sistem linear A*c = b
# ==========================================================

A = np.zeros((n_interior, n_interior))
b = np.zeros(n_interior)

for i in range(n_interior):
    A[i, i] = koef_tengah

    if i > 0:
        A[i, i - 1] = koef_kiri

    if i < n_interior - 1:
        A[i, i + 1] = koef_kanan

# Pengaruh boundary c(0) dan c(10)
b[0] = -koef_kiri * c_awal
b[-1] = -koef_kanan * c_akhir


# ==========================================================
# Penyelesaian
# ==========================================================

c_interior = np.linalg.solve(A, b)

c = np.zeros(len(x))
c[0] = c_awal
c[-1] = c_akhir
c[1:-1] = c_interior

residual = np.linalg.norm(np.dot(A, c_interior) - b, ord=np.inf)


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.27 ===")

print("\nParameter:")
print("D  =", D)
print("U  =", U)
print("k  =", k)
print("dx =", dx)
print("c(0)  =", c_awal)
print("c(10) =", c_akhir)

print("\nKoefisien beda hingga:")
print("Koefisien kiri   =", koef_kiri)
print("Koefisien tengah =", koef_tengah)
print("Koefisien kanan  =", koef_kanan)

print("\nMatriks A:")
print(np.round(A, 6))

print("\nVektor b:")
print(np.round(b, 6))

print("\nHasil konsentrasi:")
for i in range(len(x)):
    print("x = {:.0f}, c = {:.6f}".format(x[i], c[i]))

print("\nResidual ||A*c - b|| = {:.10e}".format(residual))


# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(x, c, marker="o", color="blue", label="Finite Difference")

plt.xlabel("Jarak x")
plt.ylabel("Konsentrasi c")
plt.title("Soal 11.27 - Distribusi Konsentrasi di Kanal")
plt.grid(True)
plt.legend()

plt.savefig("soal11_27.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nGrafik berhasil disimpan sebagai: soal11_27.png")
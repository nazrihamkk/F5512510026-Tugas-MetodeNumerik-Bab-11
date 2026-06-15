"""
soal11_17.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.17
Diberikan sistem nonlinear:

f(x, y) = 4 - y - 2x^2
g(x, y) = 8 - y^2 - 4x

(a) Menentukan nilai x dan y yang memenuhi kedua persamaan.
(b) Menggunakan range initial guesses x = -6 sampai 6 dan y = -6 sampai 6
    untuk menentukan initial guess yang menghasilkan setiap solusi.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def equations(variabel):
    x, y = variabel

    f = 4 - y - 2 * x**2
    g = 8 - y**2 - 4 * x

    return [f, g]


# ==========================================================
# Mencari solusi dengan beberapa tebakan awal
# ==========================================================

tebakan_awal = [
    [-6, -6], [-3, -3], [-1, 3],
    [1, 2], [2, -1], [4, -4]
]

solusi = []

for tebakan in tebakan_awal:
    akar = fsolve(equations, tebakan)

    x_akar = akar[0]
    y_akar = akar[1]

    residual = np.max(np.abs(equations([x_akar, y_akar])))

    if residual < 1e-8:
        solusi_baru = True

        for s in solusi:
            if np.linalg.norm(akar - s) < 1e-6:
                solusi_baru = False

        if solusi_baru:
            solusi.append(akar)

solusi = sorted(solusi, key=lambda v: v[0])


# ==========================================================
# Pemetaan initial guess dari -6 sampai 6
# ==========================================================

hasil_tebakan = {}

for x0 in range(-6, 7):
    for y0 in range(-6, 7):
        akar = fsolve(equations, [x0, y0])

        residual = np.max(np.abs(equations(akar)))

        if residual < 1e-8:
            for i, s in enumerate(solusi):
                if np.linalg.norm(akar - s) < 1e-5:
                    hasil_tebakan[(x0, y0)] = i + 1


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.17 ===")
print("Sistem nonlinear:")
print("f(x, y) = 4 - y - 2x^2")
print("g(x, y) = 8 - y^2 - 4x")

print("\n(a) Solusi yang memenuhi kedua persamaan:")

for i, s in enumerate(solusi, start=1):
    x = s[0]
    y = s[1]

    f_val, g_val = equations([x, y])

    print("\nSolusi {}:".format(i))
    print("x = {:.6f}".format(x))
    print("y = {:.6f}".format(y))
    print("f(x, y) = {:.2e}".format(f_val))
    print("g(x, y) = {:.2e}".format(g_val))

print("\n(b) Pengaruh initial guess x = -6 sampai 6 dan y = -6 sampai 6")
print("Jumlah initial guess yang konvergen ke tiap solusi:")

for i in range(1, len(solusi) + 1):
    jumlah = sum(1 for nilai in hasil_tebakan.values() if nilai == i)
    print("Solusi {} diperoleh dari {} initial guess".format(i, jumlah))


# ==========================================================
# Grafik
# ==========================================================

x_plot = np.linspace(-3, 3, 400)

# Dari f(x,y) = 0
# 4 - y - 2x^2 = 0  ->  y = 4 - 2x^2
y_f = 4 - 2 * x_plot**2

# Dari g(x,y) = 0
# 8 - y^2 - 4x = 0  ->  x = 2 - y^2/4
y_plot = np.linspace(-5, 5, 400)
x_g = 2 - y_plot**2 / 4

plt.figure(figsize=(8, 6))

plt.plot(x_plot, y_f, label="f(x,y)=0  ->  y = 4 - 2x^2")
plt.plot(x_g, y_plot, label="g(x,y)=0  ->  x = 2 - y^2/4")

for i, s in enumerate(solusi, start=1):
    plt.plot(s[0], s[1], "ro")
    plt.text(s[0] + 0.08, s[1] + 0.08, "S{}".format(i))

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

plt.title("Soal 11.17 - Sistem Nonlinear")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.savefig("soal11_17.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nGrafik berhasil disimpan sebagai: soal11_17.png")
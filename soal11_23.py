"""
soal11_23.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.23
Membuat analisis jumlah operasi untuk Thomas Algorithm,
kemudian membandingkannya dengan eliminasi Gauss tanpa pivoting.

Diminta:
Membuat plot jumlah operasi terhadap n dari n = 2 sampai n = 20
untuk kedua metode.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Fungsi jumlah operasi
# ==========================================================

def operasi_gauss(n):
    """
    Jumlah operasi eliminasi Gauss tanpa pivoting.
    Rumus lengkap:
    (2n^3 + 3n^2 - 5n) / 3
    """
    return (2 * n**3 + 3 * n**2 - 5 * n) / 3


def operasi_thomas(n):
    """
    Jumlah operasi Thomas Algorithm untuk sistem tridiagonal.
    Perkiraan operasi linear:
    8n - 7
    """
    return 8 * n - 7


# ==========================================================
# Data n dari 2 sampai 20
# ==========================================================

n_values = np.arange(2, 21)

gauss_values = operasi_gauss(n_values)
thomas_values = operasi_thomas(n_values)


# ==========================================================
# Output tabel
# ==========================================================

print("=== SOAL 11.23 ===")
print("Perbandingan jumlah operasi:")
print("Gauss Elimination vs Thomas Algorithm")

print("\nRumus:")
print("Gauss Elimination = (2n^3 + 3n^2 - 5n) / 3")
print("Thomas Algorithm  = 8n - 7")

print("\nTabel jumlah operasi:")
print("{:<5} {:>15} {:>15} {:>15}".format("n", "Gauss", "Thomas", "Rasio"))

for i in range(len(n_values)):
    n = n_values[i]
    gauss = gauss_values[i]
    thomas = thomas_values[i]
    rasio = gauss / thomas

    print("{:<5} {:>15.2f} {:>15.2f} {:>14.2f}x".format(
        n, gauss, thomas, rasio
    ))


# ==========================================================
# Grafik
# ==========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    n_values,
    gauss_values,
    marker="o",
    label="Gauss Elimination"
)

plt.plot(
    n_values,
    thomas_values,
    marker="s",
    label="Thomas Algorithm"
)

plt.xlabel("Ukuran sistem (n)")
plt.ylabel("Jumlah operasi")
plt.title("Soal 11.23 - Perbandingan Operasi Gauss vs Thomas")
plt.grid(True)
plt.legend()

plt.savefig("soal11_23.png", dpi=200, bbox_inches="tight")
plt.close()


# ==========================================================
# Kesimpulan
# ==========================================================

print("\nGrafik berhasil disimpan sebagai: soal11_23.png")

print("\nKesimpulan:")
print("Eliminasi Gauss memiliki orde O(n^3), sehingga jumlah operasi naik sangat cepat.")
print("Thomas Algorithm memiliki orde O(n), sehingga jumlah operasi naik secara linear.")
print("Untuk sistem tridiagonal, Thomas Algorithm jauh lebih efisien.")

print("\nUntuk n = 20:")
print("Operasi Gauss  = {:.0f}".format(operasi_gauss(20)))
print("Operasi Thomas = {:.0f}".format(operasi_thomas(20)))
print("Rasio Gauss/Thomas = {:.2f}x".format(operasi_gauss(20) / operasi_thomas(20)))
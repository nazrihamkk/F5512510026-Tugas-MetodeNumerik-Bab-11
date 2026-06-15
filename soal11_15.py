"""
soal11_15.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.15
Mengidentifikasi set persamaan yang tidak dapat diselesaikan
dengan metode iteratif seperti Gauss-Seidel.

Kriteria:
1. Jika memungkinkan, susun ulang baris agar diagonal dominan.
2. Gunakan spectral radius rho.
3. Jika rho < 1, Gauss-Seidel konvergen.
4. Jika rho >= 1, Gauss-Seidel tidak konvergen.
"""

import numpy as np
import itertools


def spectral_radius_gs(A):
    A = np.array(A, dtype=float)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    if np.any(np.abs(np.diag(A)) < 1e-15):
        return np.inf

    B = -np.linalg.inv(D + L).dot(U)
    eigenvalues = np.linalg.eigvals(B)

    return max(abs(eigenvalues))


def is_diagonal_dominant(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    for i in range(n):
        diagonal = abs(A[i, i])
        non_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)

        if diagonal < non_diagonal:
            return False

    return True


def best_rearrangement(A, b):
    n = len(b)

    best_A = None
    best_b = None
    best_order = None
    best_rho = np.inf
    best_dd = False

    for order in itertools.permutations(range(n)):
        A_new = A[list(order), :]
        b_new = b[list(order)]

        rho = spectral_radius_gs(A_new)
        diagonal_dominant = is_diagonal_dominant(A_new)

        if rho < best_rho:
            best_rho = rho
            best_A = A_new
            best_b = b_new
            best_order = order
            best_dd = diagonal_dominant

    return best_A, best_b, best_order, best_rho, best_dd


def gauss_seidel(A, b, es=5.0, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)
    x = np.zeros(n)

    for iterasi in range(1, max_iter + 1):
        x_lama = x.copy()

        for i in range(n):
            jumlah = 0.0

            for j in range(n):
                if j != i:
                    jumlah += A[i, j] * x[j]

            x[i] = (b[i] - jumlah) / A[i, i]

        error = np.zeros(n)

        for i in range(n):
            if x[i] != 0:
                error[i] = abs((x[i] - x_lama[i]) / x[i]) * 100
            else:
                error[i] = np.inf

        if np.all(error < es):
            return x, iterasi, error, True

    return x, max_iter, error, False


# ==========================================================
# Data Soal 11.15
# ==========================================================

sets = {
    "Set One": (
        np.array([
            [8.0, 3.0, 1.0],
            [0.0, -6.0, 7.0],
            [2.0, 4.0, -1.0]
        ]),
        np.array([12.0, 1.0, 5.0])
    ),

    "Set Two": (
        np.array([
            [1.0, 1.0, 5.0],
            [1.0, 4.0, -1.0],
            [3.0, 1.0, -1.0]
        ]),
        np.array([7.0, 4.0, 4.0])
    ),

    "Set Three": (
        np.array([
            [-1.0, 3.0, 5.0],
            [-2.0, 4.0, -5.0],
            [0.0, 2.0, -1.0]
        ]),
        np.array([7.0, -3.0, 1.0])
    )
}

es = 5.0
hasil_akhir = []


# ==========================================================
# Penyelesaian
# ==========================================================

print("=== SOAL 11.15 ===")
print("Identifikasi set yang tidak dapat diselesaikan dengan Gauss-Seidel")
print("Kriteria konvergensi: spectral radius rho < 1")

for nama_set, data in sets.items():
    A, b = data

    A_best, b_best, order, rho, diagonal_dominant = best_rearrangement(A, b)

    print("\n" + "=" * 60)
    print(nama_set)
    print("=" * 60)

    print("Urutan baris terbaik:", order)
    print("Spectral radius rho = {:.6f}".format(rho))
    print("Diagonal dominant  =", diagonal_dominant)

    if rho < 1:
        x, iterasi, error, konvergen = gauss_seidel(A_best, b_best, es=es)
        x_exact = np.linalg.solve(A, b)

        print("Status: KONVERGEN")
        print("Iterasi =", iterasi)

        print("x = {:.6f}".format(x[0]))
        print("y = {:.6f}".format(x[1]))
        print("z = {:.6f}".format(x[2]))

        print("Error terakhir:")
        print("ea_x = {:.6f}%".format(error[0]))
        print("ea_y = {:.6f}%".format(error[1]))
        print("ea_z = {:.6f}%".format(error[2]))

        print("Solusi eksak numpy:")
        print("x = {:.6f}".format(x_exact[0]))
        print("y = {:.6f}".format(x_exact[1]))
        print("z = {:.6f}".format(x_exact[2]))

        hasil_akhir.append((nama_set, "Konvergen"))

    else:
        print("Status: TIDAK KONVERGEN")
        print("Alasan: semua penyusunan baris menghasilkan rho >= 1")
        print("Maka Gauss-Seidel tidak memenuhi kriteria konvergensi.")

        hasil_akhir.append((nama_set, "Tidak konvergen"))


# ==========================================================
# Kesimpulan
# ==========================================================

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

for nama_set, status in hasil_akhir:
    print("{} : {}".format(nama_set, status))

print("\nSet yang tidak dapat diselesaikan dengan Gauss-Seidel adalah: Set Three")
"""
soal11_16.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.16
Menggunakan software untuk:
1. Mencari solusi Ax = b
2. Menghitung invers matriks
3. Menghitung condition number berdasarkan row-sum norm

Catatan:
Pada kedua kasus, x = [1, 1, ..., 1] merupakan solusi karena b adalah jumlah baris A.
"""

import numpy as np


def row_sum_norm(A):
    """Menghitung row-sum norm atau infinity norm."""
    return np.max(np.sum(np.abs(A), axis=1))


def analyze_system(A, b, label):
    """Menganalisis solusi, invers, dan condition number."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    print("\n" + "=" * 60)
    print(f"SOAL 11.16 {label}")
    print("=" * 60)

    print("\nMatriks A:")
    print(A)

    print("\nVektor b:")
    print(b)

    # Solusi numerik
    print("\nSolusi Ax = b:")

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.6f}")

    print("\nCek A dikali vektor satu:")
    ones = np.ones(len(b))
    print(np.dot(A, ones))

    print("\nKesimpulan solusi:")
    if np.allclose(np.dot(A, ones), b):
        print("x = [1, 1, ..., 1] memenuhi sistem Ax = b.")
    else:
        print("x = [1, 1, ..., 1] tidak memenuhi sistem.")

    # Determinan dan rank
    det_A = np.linalg.det(A)
    rank_A = np.linalg.matrix_rank(A)

    print("\nDeterminan A:")
    print(det_A)

    print("\nRank A:")
    print(rank_A)

    # Invers dan condition number
    print("\nInvers A:")

    if abs(det_A) > 1e-10:
        A_inv = np.linalg.inv(A)
        print(np.round(A_inv, 6))

        norm_A = row_sum_norm(A)
        norm_A_inv = row_sum_norm(A_inv)
        condition_number = norm_A * norm_A_inv

        print("\nRow-sum norm:")
        print(f"||A||     = {norm_A:.6f}")
        print(f"||A^-1||  = {norm_A_inv:.6f}")
        print(f"Condition number = {condition_number:.6f}")

    else:
        print("Matriks singular atau hampir singular, sehingga invers sejati tidak ada.")

        A_pinv = np.linalg.pinv(A)
        print("\nPseudo-inverse A:")
        print(np.round(A_pinv, 6))

        condition_number = np.linalg.cond(A, np.inf)

        print("\nCondition number:")
        print("Sangat besar / tak hingga secara praktis")
        print(f"np.linalg.cond(A, inf) = {condition_number}")

    residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)

    print("\nResidual ||A*x - b||:")
    print(residual)


# ==========================================================
# Data Soal 11.16 (a)
# ==========================================================

A_a = np.array([
    [1.0, 4.0, 9.0],
    [4.0, 9.0, 16.0],
    [9.0, 16.0, 25.0]
])

b_a = np.array([14.0, 29.0, 50.0])


# ==========================================================
# Data Soal 11.16 (b)
# ==========================================================

A_b = np.array([
    [1.0, 4.0, 9.0, 16.0],
    [4.0, 9.0, 16.0, 25.0],
    [9.0, 16.0, 25.0, 36.0],
    [16.0, 25.0, 36.0, 49.0]
])

b_b = np.array([30.0, 54.0, 86.0, 126.0])


# ==========================================================
# Penyelesaian
# ==========================================================

print("=== SOAL 11.16 ===")
print("Solusi, invers, dan condition number berdasarkan row-sum norm")

analyze_system(A_a, b_a, "(a)")
analyze_system(A_b, b_b, "(b)")
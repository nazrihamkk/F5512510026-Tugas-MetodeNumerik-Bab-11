"""
soal11_21.py
Nama  : Nazril Ilham Kakame
NIM   : F5512510026
Kelas : Teknik Informatika A

Soal 11.21
Diberikan matriks persegi A.
Tuliskan satu perintah MATLAB untuk membentuk matriks augmented [A | I],
yaitu matriks A yang digabung dengan matriks identitas I.

Perintah MATLAB:
Aug = [A eye(size(A,1))]

Implementasi Python:
Aug = np.hstack((A, np.eye(A.shape[0])))
"""

import numpy as np


# ==========================================================
# Contoh Matriks A
# ==========================================================

A = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])


# ==========================================================
# Single-line Augmented Matrix [A | I]
# ==========================================================

Aug = np.hstack((A, np.eye(A.shape[0])))


# ==========================================================
# Output
# ==========================================================

print("=== SOAL 11.21 ===")

print("\nPerintah MATLAB satu baris:")
print("Aug = [A eye(size(A,1))]")

print("\nPerintah Python ekuivalen:")
print("Aug = np.hstack((A, np.eye(A.shape[0])))")

print("\nMatriks A:")
print(A)

print("\nMatriks Identitas I:")
print(np.eye(A.shape[0]))

print("\nMatriks Augmented [A | I]:")
print(Aug)
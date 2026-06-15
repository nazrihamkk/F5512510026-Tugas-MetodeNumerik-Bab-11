# Tugas Metode Numerik - Bab 11: Special Matrices & Gauss-Seidel

<pre>
Nama        : Nazril Ilham Kakame
NIM         : F5512510026
Kelas       : Teknik Informatika A
Mata Kuliah : Metode Numerik
Topik       : Bab 11 - Special Matrices dan Gauss-Seidel
</pre>

Repository ini berisi penyelesaian soal **11.1 sampai 11.28** menggunakan bahasa pemrograman **Python**. 
---

## Daftar Isi

- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Library yang Digunakan](#library-yang-digunakan)
- [Struktur File](#struktur-file)
- [File Output Grafik](#file-output-grafik)
- [Ringkasan Setiap Soal](#ringkasan-setiap-soal)
- [Ringkasan Metode Numerik](#ringkasan-metode-numerik)
- [Catatan Penting](#catatan-penting)

---

## Cara Menjalankan Program

### 1. Install library

```bash
pip install numpy scipy matplotlib
```

### 2. Jalankan satu file soal

Contoh:

```bash
python soal11_1.py
python soal11_2.py
python soal11_3.py
```

### 3. Jalankan semua file sekaligus

#### Windows Command Prompt

```bat
for %f in (soal11_*.py) do python %f
```

#### Linux / macOS

```bash
for f in soal11_*.py; do python "$f"; done
```

---

## Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `numpy` | Operasi array, matriks, sistem persamaan linear, invers, norm, condition number |
| `scipy` | Solver nonlinear pada soal tertentu, terutama `fsolve` |
| `matplotlib` | Membuat grafik untuk visualisasi hasil |

---

## Struktur File

| File | Keterangan |
|---|---|
| `soal11_1.py` | Thomas Algorithm dan Gauss-Seidel untuk sistem tridiagonal |
| `soal11_2.py` | Invers matriks dengan LU Decomposition dan unit vectors |
| `soal11_3.py` | Thomas Algorithm untuk sistem tridiagonal Crank-Nicolson |
| `soal11_4.py` | Validasi Cholesky Decomposition |
| `soal11_5.py` | Penyelesaian sistem simetris dengan Cholesky |
| `soal11_6.py` | Penyelesaian sistem simetris dengan Cholesky |
| `soal11_7.py` | Analisis Cholesky pada matriks diagonal |
| `soal11_8.py` | Gauss-Seidel dengan overrelaxation |
| `soal11_9.py` | Gauss-Seidel untuk sistem konsentrasi reaktor |
| `soal11_10.py` | Jacobi Iteration untuk sistem konsentrasi reaktor |
| `soal11_11.py` | Gauss-Seidel sampai error relatif kurang dari 5% |
| `soal11_12.py` | Gauss-Seidel tanpa dan dengan underrelaxation |
| `soal11_13.py` | Gauss-Seidel tanpa dan dengan overrelaxation |
| `soal11_14.py` | Analisis grafis Gauss-Seidel untuk slope 1 dan -1 |
| `soal11_15.py` | Identifikasi set persamaan yang tidak konvergen |
| `soal11_16.py` | Solusi, invers, dan condition number |
| `soal11_17.py` | Sistem nonlinear simultan |
| `soal11_18.py` | Aplikasi produksi komponen elektronik |
| `soal11_19.py` | Hilbert matrix 10 dimensi dan condition number |
| `soal11_20.py` | Vandermonde matrix 6 dimensi dan condition number |
| `soal11_21.py` | Perintah satu baris augmented matrix `[A | I]` |
| `soal11_22.py` | Bentuk matriks, solusi, transpose, dan invers |
| `soal11_23.py` | Analisis operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_24.py` | Program user-friendly Thomas Algorithm |
| `soal11_25.py` | Program user-friendly Cholesky Decomposition |
| `soal11_26.py` | Program user-friendly Gauss-Seidel |
| `soal11_27.py` | Finite Difference untuk kanal satu dimensi |
| `soal11_28.py` | Solver sistem pentadiagonal bandwidth 5 |

---

## File Output Grafik

Beberapa file akan menghasilkan grafik otomatis ketika dijalankan.

| File Grafik | Dihasilkan Oleh | Keterangan |
|---|---|---|
| `soal11_14.png` | `soal11_14.py` | Grafik garis slope 1 dan -1 serta jalur iterasi Gauss-Seidel |
| `soal11_17.png` | `soal11_17.py` | Grafik sistem nonlinear dan titik solusi |
| `soal11_23.png` | `soal11_23.py` | Grafik perbandingan operasi Gauss Elimination dan Thomas Algorithm |
| `soal11_27.png` | `soal11_27.py` | Grafik profil konsentrasi kanal satu dimensi |

---

# Ringkasan Setiap Soal

## Soal 11.1 - Thomas Algorithm dan Gauss-Seidel

**File:** `soal11_1.py`

Soal ini menyelesaikan sistem tridiagonal 3x3:

```text
 0.8x1 - 0.4x2        = 41
-0.4x1 + 0.8x2 -0.4x3 = 25
        -0.4x2 +0.8x3 = 105
```

Metode yang digunakan:

1. **Thomas Algorithm** untuk solusi langsung sistem tridiagonal.
2. **Gauss-Seidel** dengan toleransi `es = 5%` untuk solusi pendekatan.

Hasil Thomas Algorithm:

```text
x1 = 173.750000
x2 = 245.000000
x3 = 253.750000
```

Hasil Gauss-Seidel:

```text
Konvergen pada iterasi ke-6
x1 = 167.871094
x2 = 239.121094
x3 = 250.810547
```

Kesimpulan: Thomas Algorithm menghasilkan solusi langsung yang sama dengan `numpy.linalg.solve`, sedangkan Gauss-Seidel menghasilkan pendekatan karena dihentikan pada toleransi 5%.

---

## Soal 11.2 - Invers Matriks dengan LU Decomposition

**File:** `soal11_2.py`

Soal ini menentukan invers matriks Example 11.1 menggunakan **LU Decomposition** dan **unit vectors**. Matriks difaktorkan menjadi:

```text
A = L U
```

Setiap kolom invers diperoleh dengan menyelesaikan:

```text
L y = e_i
U x = y
```

Hasil invers:

```text
[[0.755841 0.541916 0.349667 0.171406]
 [0.541916 1.105509 0.713322 0.349667]
 [0.349667 0.713322 1.105509 0.541916]
 [0.171406 0.349667 0.541916 0.755841]]
```

Verifikasi:

```text
A * A_inv = I
Max error terhadap numpy.linalg.inv = 5.551115123125783e-17
```

---

## Soal 11.3 - Thomas Algorithm untuk Crank-Nicolson

**File:** `soal11_3.py`

Soal ini menyelesaikan sistem tridiagonal 4x4 dari skema Crank-Nicolson menggunakan Thomas Algorithm.

Data diagonal:

```text
Sub-diagonal   = -0.020875
Diagonal utama = 2.01475
Super-diagonal = -0.020875
RHS            = [4.175, 0, 0, 2.0875]
```

Hasil:

```text
T1 = 2.072441
T2 = 0.021586
T3 = 0.010960
T4 = 1.036222
```

Verifikasi dengan `numpy.linalg.solve` menghasilkan nilai yang sama.

---

## Soal 11.4 - Validasi Cholesky Decomposition

**File:** `soal11_4.py`

Soal ini memvalidasi dekomposisi Cholesky dari Example 11.2.

Matriks:

```text
A = [[6, 15, 55],
     [15, 55, 225],
     [55, 225, 979]]
```

Hasil Cholesky:

```text
L = [[ 2.449490,  0.000000, 0.000000],
     [ 6.123724,  4.183300, 0.000000],
     [22.453656, 20.916501, 6.110101]]
```

Verifikasi:

```text
L * L^T = A
Max error = 8.8817841970e-16
```

---

## Soal 11.5 - Cholesky untuk Sistem Simetris

**File:** `soal11_5.py`

Soal ini menyelesaikan sistem simetris menggunakan Cholesky Decomposition.

Langkah penyelesaian:

1. Faktorkan `A = L L^T`.
2. Selesaikan `L y = b`.
3. Selesaikan `L^T a = y`.

Hasil:

```text
a0 = 2.478571
a1 = 2.359286
a2 = 1.860714
```

Verifikasi:

```text
A * a = [152.6, 585.6, 2488.8]
Residual = 0.0
```

---

## Soal 11.6 - Penyelesaian Sistem Simetris dengan Cholesky

**File:** `soal11_6.py`

Matriks sistem:

```text
A = [[ 8, 20, 15],
     [20, 80, 50],
     [15, 50, 60]]
```

Vektor ruas kanan:

```text
b = [50, 250, 100]
```

Hasil:

```text
x1 = -2.734375
x2 =  4.882812
x3 = -1.718750
```

Verifikasi:

```text
A * x = [50, 250, 100]
Residual = 1.4210854715e-14
```

---

## Soal 11.7 - Cholesky pada Matriks Diagonal

**File:** `soal11_7.py`

Matriks diagonal:

```text
A = [[9, 0, 0],
     [0, 25, 0],
     [0, 0, 4]]
```

Hasil Cholesky:

```text
L = [[3, 0, 0],
     [0, 5, 0],
     [0, 0, 2]]
```

Kesimpulan: untuk matriks diagonal positif, Cholesky menghasilkan matriks diagonal baru yang elemen diagonalnya adalah akar dari elemen diagonal matriks asal.

---

## Soal 11.8 - Gauss-Seidel dengan Overrelaxation

**File:** `soal11_8.py`

Soal ini menyelesaikan sistem tridiagonal dari Soal 11.1 menggunakan Gauss-Seidel dengan faktor relaksasi:

```text
lambda = 1.2
es = 5%
```

Hasil:

```text
Konvergen pada iterasi ke-4
x1 = 171.422976
x2 = 244.388659
x3 = 253.622200
```

Overrelaxation mempercepat konvergensi dibanding Gauss-Seidel biasa pada sistem ini.

---

## Soal 11.9 - Gauss-Seidel untuk Reaktor Tersambung

**File:** `soal11_9.py`

Sistem:

```text
15c1 - 3c2 - c3 = 3800
-3c1 + 18c2 - 6c3 = 1200
-4c1 - c2 + 12c3 = 2350
```

Hasil Gauss-Seidel:

```text
Konvergen pada iterasi ke-4
c1 = 319.325357 g/m^3
c2 = 226.540187 g/m^3
c3 = 321.153468 g/m^3
```

Solusi eksak:

```text
c1 = 320.207254
c2 = 227.202073
c3 = 321.502591
```

---

## Soal 11.10 - Jacobi Iteration

**File:** `soal11_10.py`

Soal ini mengulang Soal 11.9, tetapi menggunakan metode Jacobi. Pada metode Jacobi, semua variabel baru dihitung menggunakan nilai lama dari iterasi sebelumnya.

Hasil:

```text
Konvergen pada iterasi ke-4
c1 = 315.285494 g/m^3
c2 = 219.066358 g/m^3
c3 = 315.621142 g/m^3
```

Kesimpulan: dibanding Gauss-Seidel, hasil Jacobi pada jumlah iterasi yang sama cenderung kurang dekat dengan solusi eksak.

---

## Soal 11.11 - Gauss-Seidel dengan Toleransi 5%

**File:** `soal11_11.py`

Sistem:

```text
10x1 + 2x2 - x3 = 27
-3x1 - 6x2 + 2x3 = -61.5
x1 + x2 + 5x3 = -21.5
```

Hasil:

```text
Konvergen pada iterasi ke-5
x1 = 0.500253
x2 = 8.000112
x3 = -6.000073
```

Solusi eksak:

```text
x1 = 0.500000
x2 = 8.000000
x3 = -6.000000
```

---

## Soal 11.12 - Gauss-Seidel dengan Underrelaxation

**File:** `soal11_12.py`

Sistem disusun ulang agar lebih mudah konvergen:

```text
6x1 - x2 - x3 = 3
6x1 + 9x2 + x3 = 40
-3x1 + x2 + 12x3 = 50
```

Hasil tanpa relaksasi:

```text
lambda = 1.0
Iterasi = 4
x1 = 1.696789
x2 = 2.829356
x3 = 4.355084
```

Hasil dengan underrelaxation:

```text
lambda = 0.95
Iterasi = 3
x1 = 1.709692
x2 = 2.830032
x3 = 4.356408
```

Solusi eksak:

```text
x1 = 1.697368
x2 = 2.828947
x3 = 4.355263
```

---

## Soal 11.13 - Gauss-Seidel dengan Overrelaxation

**File:** `soal11_13.py`

Sistem disusun ulang menjadi:

```text
-8x1 + x2 - 2x3 = -20
2x1 - 6x2 - x3 = -38
-3x1 - x2 + 7x3 = -34
```

Hasil tanpa relaksasi:

```text
lambda = 1.0
Iterasi = 3
x1 = 4.004659
x2 = 7.991680
x3 = -1.999192
```

Hasil dengan overrelaxation:

```text
lambda = 1.2
Iterasi = 6
x1 = 4.012225
x2 = 8.027261
x3 = -1.979007
```

Solusi eksak:

```text
x1 = 4.000000
x2 = 8.000000
x3 = -2.000000
```

---

## Soal 11.14 - Analisis Grafis Slope 1 dan -1

**File:** `soal11_14.py`  
**Output grafik:** `soal11_14.png`

Sistem:

```text
x1 + x2 = 3
x1 - x2 = 1
```

Solusi eksak:

```text
x1 = 2
x2 = 1
```

Iterasi Gauss-Seidel dari titik awal `(0, 0)` menghasilkan pola:

```text
(0, 0) -> (3, 2) -> (1, 0) -> (3, 2) -> (1, 0) -> ...
```

Kesimpulan: metode Gauss-Seidel berosilasi dan tidak konvergen menuju solusi eksak.

---

## Soal 11.15 - Identifikasi Set Tidak Konvergen

**File:** `soal11_15.py`

Tiga set sistem persamaan diuji dengan penyusunan ulang baris dan spectral radius Gauss-Seidel.

Hasil:

```text
Set One   : Konvergen
Set Two   : Konvergen
Set Three : Tidak konvergen
```

Kesimpulan: set yang tidak dapat diselesaikan dengan Gauss-Seidel adalah **Set Three**.

---

## Soal 11.16 - Solusi, Invers, dan Condition Number

**File:** `soal11_16.py`

Soal ini menghitung:

1. Solusi sistem `Ax = b`.
2. Invers matriks jika tersedia.
3. Condition number menggunakan row-sum norm.

Bagian (a):

```text
x = [1, 1, 1]
Condition number = 750
```

Bagian (b): matriks sangat buruk atau singular secara numerik. Program mengecek determinan dan rank, lalu menggunakan pseudo-inverse bila invers sejati tidak dapat digunakan dengan aman.

---

## Soal 11.17 - Sistem Nonlinear Simultan

**File:** `soal11_17.py`  
**Output grafik:** `soal11_17.png`

Sistem:

```text
f(x, y) = 4 - y - 2x^2
g(x, y) = 8 - y^2 - 4x
```

Solusi real yang ditemukan:

```text
(-2.000000, -4.000000)
(-0.618034, 3.236068)
(1.000000, 2.000000)
(1.618034, -1.236068)
```

Metode `fsolve` digunakan dengan beberapa tebakan awal untuk menemukan semua titik potong kurva.

---

## Soal 11.18 - Produksi Komponen Elektronik

**File:** `soal11_18.py`

Model sistem:

```text
4T + 3R + 2C = 960
T + 3R + C = 510
2T + R + 3C = 610
```

Hasil produksi:

```text
Transistor = 120 unit
Resistor = 100 unit
Computer chips = 90 unit
```

Verifikasi menunjukkan semua material digunakan tepat sesuai persediaan.

---

## Soal 11.19 - Hilbert Matrix 10 Dimensi

**File:** `soal11_19.py`

Hilbert matrix merupakan contoh klasik matriks yang sangat ill-conditioned.

Hasil:

```text
Spectral condition number = 1.602498e+13
Estimasi digit hilang = 13.20 digit
Max error = sekitar 4.517e-4
```

Kesimpulan: meskipun solusi eksak adalah semua `x = 1`, hasil numerik memiliki penyimpangan kecil akibat round-off error dan condition number yang sangat besar.

---

## Soal 11.20 - Vandermonde Matrix 6 Dimensi

**File:** `soal11_20.py`

Data:

```text
x = [4, 2, 7, 10, 3, 5]
```

Hasil:

```text
Spectral condition number = 1.449173e+07
Estimasi digit hilang = 7.16 digit
Max error = sekitar 6.0168e-11
```

Kesimpulan: matriks Vandermonde cukup ill-conditioned, tetapi hasil numerik masih sangat dekat dengan solusi eksak `c = [1, 1, 1, 1, 1, 1]`.

---

## Soal 11.21 - Augmented Matrix `[A | I]`

**File:** `soal11_21.py`

Soal ini meminta satu baris perintah MATLAB untuk membentuk matriks augmented `[A | I]`.

Jawaban MATLAB:

```matlab
Aug = [A eye(size(A,1))]
```

Ekuivalen Python:

```python
Aug = np.hstack((A, np.eye(A.shape[0])))
```

---

## Soal 11.22 - Bentuk Matriks, Transpose, dan Invers

**File:** `soal11_22.py`

Persamaan:

```text
50 = 5x3 - 7x2
4x2 + 7x3 + 30 = 0
x1 - 7x3 = 40 - 3x2 + 5x1
```

Bentuk standar:

```text
0x1 - 7x2 + 5x3 = 50
0x1 + 4x2 + 7x3 = -30
-4x1 + 3x2 - 7x3 = 40
```

Hasil:

```text
x1 = -15.181159
x2 = -7.246377
x3 = -0.144928
```

Program juga menampilkan transpose dan invers matriks koefisien.

---

## Soal 11.23 - Analisis Operasi Thomas vs Gauss

**File:** `soal11_23.py`  
**Output grafik:** `soal11_23.png`

Rumus operasi:

```text
Gauss Elimination = (2n^3 + 3n^2 - 5n) / 3
Thomas Algorithm = 8n - 7
```

Untuk `n = 20`:

```text
Operasi Gauss = 5700
Operasi Thomas = 153
Rasio = 37.25x
```

Kesimpulan: Thomas Algorithm jauh lebih efisien untuk sistem tridiagonal karena kompleksitasnya linear.

---

## Soal 11.24 - Program User-Friendly Thomas Algorithm

**File:** `soal11_24.py`

Program membuat fungsi umum `thomas_algorithm()` untuk menyelesaikan sistem tridiagonal.

Uji Example 11.1 menghasilkan:

```text
x1 = 65.969834
x2 = 93.778462
x3 = 124.538228
x4 = 159.479524
```

Verifikasi dengan NumPy memberikan hasil sama.

---

## Soal 11.25 - Program User-Friendly Cholesky Decomposition

**File:** `soal11_25.py`

Program membuat fungsi umum `cholesky_decomposition()` untuk menghitung:

```text
A = L * L^T
```

Uji Example 11.2 menghasilkan:

```text
L = [[ 2.449490,  0.000000, 0.000000],
     [ 6.123724,  4.183300, 0.000000],
     [22.453656, 20.916501, 6.110101]]
```

Verifikasi:

```text
L * L^T = A
Max error = 8.8817841970e-16
```

---

## Soal 11.26 - Program User-Friendly Gauss-Seidel

**File:** `soal11_26.py`

Program membuat fungsi umum `gauss_seidel()` dengan parameter:

```text
A, b, es, max_iter, lambda
```

Uji Example 11.3 menghasilkan:

```text
Konvergen pada iterasi ke-3
x1 = 3.000032
x2 = -2.499988
x3 = 6.999999
```

Solusi eksak:

```text
x1 = 3.000000
x2 = -2.500000
x3 = 7.000000
```

---

## Soal 11.27 - Finite Difference pada Kanal Satu Dimensi

**File:** `soal11_27.py`  
**Output grafik:** `soal11_27.png`

Persamaan:

```text
0 = D d²c/dx² - U dc/dx - kc
```

Parameter:

```text
D = 2
U = 1
k = 0.2
c(0) = 80
c(10) = 20
dx = 2
```

Koefisien beda hingga:

```text
kiri   = 0.75
tengah = -1.20
kanan  = 0.25
```

Hasil konsentrasi:

```text
x = 0,  c = 80.000000
x = 2,  c = 59.101256
x = 4,  c = 43.686027
x = 6,  c = 32.389161
x = 8,  c = 24.409892
x = 10, c = 20.000000
```

---

## Soal 11.28 - Solver Pentadiagonal Bandwidth 5

**File:** `soal11_28.py`

Sistem pentadiagonal:

```text
[ 8  -2  -1   0   0 ] [x1]   [5]
[-2   9  -4  -1   0 ] [x2]   [2]
[-1  -3   7  -1  -2 ] [x3] = [0]
[ 0  -4  -2  12  -5 ] [x4]   [1]
[ 0   0  -7  -3  15 ] [x5]   [5]
```

Hasil:

```text
x1 = 1.000000
x2 = 1.000000
x3 = 1.000000
x4 = 1.000000
x5 = 1.000000
```

Verifikasi:

```text
A * x = [5, 2, 0, 1, 5]
Residual = 3.5527136788e-15
```

---

# Ringkasan Metode Numerik

| Kelompok Soal | Metode Utama | Keterangan |
|---|---|---|
| 11.1, 11.3, 11.24 | Thomas Algorithm | Efisien untuk sistem tridiagonal, kompleksitas O(n) |
| 11.2 | LU Decomposition | Digunakan untuk mencari invers matriks |
| 11.4-11.7, 11.25 | Cholesky Decomposition | Digunakan untuk matriks simetris positif definit |
| 11.8-11.13, 11.26 | Gauss-Seidel dan Relaksasi | Metode iteratif dengan toleransi error |
| 11.10 | Jacobi Iteration | Metode iteratif menggunakan nilai lama semua variabel |
| 11.14 | Analisis Grafis | Menunjukkan osilasi atau ketidak-konvergenan |
| 11.15 | Spectral Radius | Menentukan konvergensi Gauss-Seidel |
| 11.16, 11.19, 11.20 | Condition Number | Menganalisis sensitivitas dan ill-conditioning |
| 11.17 | Solver Nonlinear | Menggunakan `fsolve` untuk sistem nonlinear |
| 11.18 | Pemodelan Sistem Linear | Aplikasi produksi elektronik |
| 11.23 | Analisis Operasi | Membandingkan jumlah operasi algoritma |
| 11.27 | Finite Difference | Mengubah persamaan diferensial menjadi sistem linear |
| 11.28 | Pentadiagonal Solver | Ekstensi Thomas Algorithm untuk bandwidth 5 |

---

# Catatan Penting

1. Semua program dibuat menggunakan Python.
2. Beberapa soal pada buku menyebut MATLAB atau Mathcad, tetapi pada tugas ini NumPy, SciPy, dan Matplotlib digunakan sebagai pengganti software numerik tersebut.
3. Hasil program diverifikasi menggunakan `numpy.linalg.solve`, residual, atau rekonstruksi matriks.
4. Nilai residual yang sangat kecil seperti `1e-14` dianggap nol secara numerik karena keterbatasan floating point.
5. Untuk metode iteratif seperti Gauss-Seidel dan Jacobi, hasil bisa berbeda sedikit dari solusi eksak karena iterasi dihentikan ketika error relatif sudah memenuhi toleransi.

---

# Kesimpulan Umum

Bab 11 membahas berbagai metode khusus untuk menyelesaikan sistem persamaan linear dan nonlinear secara efisien. Sistem dengan struktur khusus seperti tridiagonal dan pentadiagonal dapat diselesaikan jauh lebih cepat dibanding eliminasi Gauss umum. Metode iteratif seperti Gauss-Seidel dan Jacobi bergantung pada kondisi konvergensi, sedangkan analisis condition number menunjukkan seberapa sensitif suatu sistem terhadap kesalahan pembulatan. Seluruh program pada repository ini dibuat untuk menunjukkan implementasi praktis dari konsep-konsep tersebut menggunakan Python.

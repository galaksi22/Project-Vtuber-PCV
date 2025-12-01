5. Metodologi & Penjelasan Teknis
Logika inti dari sistem ini bergantung pada perhitungan rasio geometri landmark wajah, bukan sekadar jarak piksel, sehingga sistem tetap akurat meskipun jarak wajah pengguna berubah-ubah terhadap kamera.

A. Eye Aspect Ratio (EAR) - Deteksi Mata
EAR digunakan untuk menentukan keterbukaan mata. Rumus yang digunakan adalah perbandingan rata-rata jarak vertikal kelopak mata terhadap jarak horizontal sudut mata.

Formula Logika:
EAR > 0.28: Diklasifikasikan sebagai Normal (Aset Mata 3).
0.20 < EAR < 0.28: Diklasifikasikan sebagai Squint (Aset Mata 2).
EAR < 0.20: Diklasifikasikan sebagai Blink (Aset Mata 1).

B. Mouth Aspect Ratio (MAR) - Deteksi Mulut
Serupa dengan EAR, MAR menghitung rasio tinggi bukaan bibir bagian dalam terhadap lebar mulut.

Formula Logika:
MAR > 0.3: Diklasifikasikan sebagai Open (Mangap).
MAR > 0.05: Diklasifikasikan sebagai Talk (Bicara).
MAR < 0.05: Diklasifikasikan sebagai Idle (Diam).

C. Head Pose Estimation (Orientasi)
Sistem menentukan orientasi wajah dengan membandingkan posisi relatif titik hidung (nose tip landmark) terhadap titik pipi kiri dan kanan pada sumbu X.
Jika rasio posisi hidung bergeser signifikan (< 0.35 atau > 0.65), sistem menganggap pengguna menoleh dan mengganti aset wajah dasar (Base Face) serta menyesuaikan perspektif mata menggunakan parameter Scale Y (efek pipih).

6. Petunjuk Instalasi & Penggunaan
   git clone [https://github.com/galaksi22/Project-Vtuber-PCV.git](https://github.com/galaksi22/Project-Vtuber-PCV.git)
cd Project-Vtuber-PCV

Langkah 2: Instalasi Dependensi
Pastikan Python telah terinstal, kemudian jalankan perintah berikut:
pip install opencv-python mediapipe numpy

Langkah 3: Menjalankan Program
python ProjectVtuber.py

7. Kontrol Program (Keyboard Shortcuts)
Saat jendela aplikasi aktif, pengguna dapat menggunakan tombol berikut untuk kontrol sistem:
Tombol,Fungsi,Deskripsi
O,OPEN Panel,Membuka jendela Control Panel untuk melakukan kalibrasi posisi aset.
X,EXIT Panel,Menutup jendela Control Panel (Mode bersih).
S,SAVE Config,Menyimpan seluruh parameter koordinat saat ini ke file JSON.
Q,QUIT,Menghentikan program dan menutup aplikasi.

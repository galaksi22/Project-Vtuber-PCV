⚙️ Cara Menjalankan
Clone Repo ini:

Bash

git clone [https://github.com/galaksi22/Project-Vtuber-PCV.git](https://github.com/galaksi22/Project-Vtuber-PCV.git)
cd Project-Vtuber-PCV
Install Library yang dibutuhkan:

Bash

pip install opencv-python mediapipe numpy
Run Program: Pastikan webcam nyala, lalu ketik:

Bash

python ProjectVtuber.py
🎮 Shortcut Keyboard
Pas program jalan, klik dulu jendelanya, terus pakai tombol ini:

O : OPEN Control Panel (Buka Slider buat geser-geser aset).

X : EXIT Control Panel (Tutup Slider biar bersih).

S : SAVE Config (Simpan settingan posisi ke JSON).

Q : QUIT (Keluar program).

🧠 Penjelasan Teknis (Metode Tracking)
Di project ini saya nggak cuma pakai deteksi wajah standar, tapi pakai perhitungan rasio biar akurat mau dekat atau jauh dari kamera.

1. Deteksi Mata (EAR - Eye Aspect Ratio)
Saya pakai rumus EAR buat ngitung rasio bukaan mata.

Logikanya: Saya ambil 6 titik koordinat di mata. Kalau jarak vertikalnya memendek dibanding jarak horizontal, berarti lagi merem.

Threshold:

EAR > 0.28 : Mata Normal (Pakai aset mata 3)

0.20 - 0.28: Mata Sayu (Pakai aset mata 2)

EAR < 0.20 : Merem (Pakai aset mata 1)

2. Deteksi Mulut (MAR - Mouth Aspect Ratio)
Mirip kayak mata, tapi buat mulut.

Threshold:

MAR > 0.3 : Mangap Lebar (Open)

MAR > 0.05: Ngomong (Talk)

MAR < 0.05: Diam (Idle)

3. Orientasi Kepala (Head Pose)
Buat nentuin arah noleh, saya bandingin posisi titik hidung (nose tip) sama pipi kiri dan kanan.

Kalau titik hidung geser drastis ke arah koordinat pipi kiri, berarti user lagi noleh kiri -> Program bakal nge-load gambar Muka_Kiri.png.

Begitu juga buat kanan, atas, dan bawah.

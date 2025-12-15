# 📂 Direktori Aset Visual (Assets)

Folder ini berisi seluruh komponen gambar yang digunakan oleh **VTuber Engine** untuk menyusun karakter secara *real-time*.

## 1. Struktur & Penamaan File
Sistem Python memuat gambar berdasarkan **nama file yang spesifik**. Jika Anda ingin mengubah karakter, **timpa (replace)** file gambar dengan nama yang sama persis, jangan mengubah nama filenya kecuali Anda juga mengubah kode program.

### A. Wajah Dasar (Base Face)
Gambar dasar kepala sesuai arah tolehan.
* `Muka_Depan.png` - Wajah menghadap depan (Default).
* `Muka_Kiri.png` - Wajah menoleh ke kiri.
* `Muka_Kanan.png` - Wajah menoleh ke kanan.
* `Muka_Atas.png` - Wajah menoleh ke atas.
* `muka_bawah.png` - Wajah menoleh ke bawah.

### B. Komponen Ekspresi
Bagian-bagian wajah yang bergerak dinamis.
* **Mata Kanan (User's Right):**
  * `Mata_Kanan_2.png` - Mata Normal (Terbuka).
  * `Mata_Kanan_3.png` - Mata Lebar/Terkejut (Wide).
  * `Mata_Kanan_1.png` - Mata Tertutup (Blink).
* **Mata Kiri (User's Left):**
  * `Mata_Kiri_2.png` - Mata Normal (Terbuka).
  * `Mata_Kiri_3.png` - Mata Lebar/Terkejut (Wide).
  * `Mata_Kiri_1.png` - Mata Tertutup (Blink).
* **Mulut:**
  * `Mulut_1.png` - Diam (Idle/Senyum tipis).
  * `Mulut_2.png` - Bicara (Talk).
  * `Mulut_3.png` - Terbuka Lebar (Open/Surprised).
  * `Mulut_4.png` - Tertawa (Laugh/Gigi terlihat).

### C. Tubuh & Gestur Tangan
* `badan_full.png` - Tubuh standar (tangan di bawah).
* `dua_tangan_naik.png` - Pose kedua tangan naik (Sorak/Hooray).
* `dua_tangan_T.png` - Pose T-Pose.
* **Variasi Tangan Kanan & Kiri:**
  * File dengan awalan `tangan_kanan_...` (misal: peace, jempol, 1-5).
  * File dengan awalan `tangan_kiri_...` (misal: peace, jempol, 1-5).

### D. Latar Belakang
* `Background.png` - Latar belakang utama.
* File gambar lain (`.jpg`, `.png`) di folder ini akan otomatis terdeteksi sebagai *background* alternatif yang bisa diganti dengan tombol `<` dan `>`.

---

## 2. Spesifikasi Teknis Gambar (Penting!)

Agar animasi berjalan lancar dan visual tidak rusak, perhatikan aturan berikut:

1.  **Format Transparan (PNG):** Semua aset karakter (Mata, Mulut, Tubuh, Wajah) **WAJIB** berformat `.png` dengan *Transparent Background* (Alpha Channel). Jangan gunakan JPG untuk karakter.
2.  **Dimensi Konsisten:** Disarankan agar `Muka_Depan.png` dan variasi arah lainnya memiliki resolusi kanvas yang sama agar transisi terlihat mulus.
3.  **Cropping:** Aset mata dan mulut sebaiknya di-*crop* pas pada objeknya (tanpa terlalu banyak ruang kosong transparan di sekitarnya) untuk memudahkan kalibrasi posisi di program.

## 3. Cara Mengganti Karakter (Custom Avatar)
1.  Siapkan gambar karakter Anda yang sudah dipecah per bagian (Mata, Mulut, Kepala, Tubuh).
2.  Simpan di folder ini.
3.  **Rename** gambar Anda sesuai daftar nama di atas (timpa file lama).
4.  Jalankan program dan gunakan menu **Editor (`Tombol O`)** untuk menyesuaikan ulang posisi mata dan mulut jika bergeser.
5.  Tekan **`S`** untuk menyimpan konfigurasi baru.
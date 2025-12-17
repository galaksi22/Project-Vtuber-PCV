# Project Akhir: Real-Time 2D VTuber Engine

**Mata Kuliah:** Pengolah Citra dan Video (PCV)  
**Topik:** Body Tracking, Gesture Recognition, Face Tracking 

---

| Identitas | Detail |
| :--- | :--- |
| **Nama** | **Alfito Ichsan Galaksi** |
| **NRP** | **5024231071** |

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sistem **Virtual YouTuber (VTuber) 2D** yang beroperasi secara *real-time* menggunakan input kamera tunggal (webcam). Sistem ini dibangun menggunakan Python dengan memanfaatkan **MediaPipe Holistic** untuk melacak wajah, tangan, dan postur tubuh secara simultan.

Berbeda dengan sistem *face-tracking* sederhana, proyek ini mengimplementasikan **Procedural Animation** dan **Hybrid Tracking Logic**. Sistem tidak hanya menempelkan gambar, tetapi juga menghitung fisika sederhana (seperti pernapasan dan kemiringan tubuh) serta menggunakan logika "Garis Merah" (*thresholding*) untuk mendeteksi gestur tangan dengan presisi tinggi tanpa memerlukan peralatan *motion capture* mahal.

## 2. Fitur Utama & Kapabilitas

### A. Core Tracking (Pelacakan)
* **Holistic Tracking:** Menggabungkan deteksi Wajah (478 landmark), Tangan (21 landmark per tangan), dan Pose Tubuh.
* **Smart Arm Detection (Logika Garis Merah):** Menggunakan referensi posisi hidung sebagai garis batas (*threshold*). Jika pergelangan tangan melewati garis hidung, karakter otomatis mengangkat tangan, mengabaikan gangguan kecil pada jari.
* **Finger Gesture Recognition:** Mendeteksi pose jari spesifik seperti: Angka 1-5, *Peace*, dan *Thumb Up*.

### B. Procedural Animation (Animasi Otomatis)
* **Dynamic Body Leaning (Sway):** Tubuh karakter miring secara halus mengikuti sudut bahu pengguna menggunakan interpolasi (*Smoothing Factor*) agar gerakan tidak kaku.
* **Auto-Breathing (Pernapasan):** Simulasi gerakan dada naik-turun menggunakan fungsi gelombang sinus (*Sine Wave*) untuk memberikan kesan "hidup" saat karakter diam.

### C. Visual & UI Enhancement
* **Estetik Glassmorphism UI:** Antarmuka menu navigasi modern dengan latar belakang gradasi transparan dan border *rounded* yang elegan.
* **Image Sharpening (Fitur K):** Implementasi filter konvolusi (kernel 3x3) untuk mempertajam visual karakter secara *real-time*.
* **Seamless Background Transition:** Transisi latar belakang yang halus (*cross-dissolve*) saat mengganti *scene*.

## 3. Spesifikasi Lingkungan Pengembangan
Proyek ini dikembangkan menggunakan pustaka berikut. Pastikan Anda telah menginstalnya:
| Pustaka / Library | Fungsi Utama |
| :--- | :--- |
| **Python 3.x** | *Runtime Environment* utama. |
| **OpenCV (`cv2`)** | Manipulasi citra, rendering visual, filter sharpening, dan UI drawing. |
| **MediaPipe** | Ekstraksi *Holistic Landmarks* (Face, Hands, Pose). |
| **NumPy** | Operasi matriks untuk gradasi warna dan perhitungan geometri vektor. |

## 4. Struktur Direktori
```text
Project-Vtuber-PCV/
│
├── assets/                 # Direktori aset visual (PNG Layering)
│   ├── badan_full.png      # Base body
│   ├── dua_tangan_naik.png # Variasi gestur
│   ├── Muka_Depan.png      # Base wajah
│   ├── Background.png      # Latar belakang default
│   └── ... (Aset mata, mulut, dan tangan lainnya)
│
├── vtuber_body.json        # Konfigurasi posisi TUBUH (Auto-save)
├── vtuber_face.json        # Konfigurasi posisi WAJAH (Auto-save)
├── ProjectVtuber.py        # Source code utama (Main Engine)
└── pose_tracking.py        # Script eksternal ini berisi logika deteksi terpisah (Opsional)
```

## 5. Metodologi & Logika Teknis
   
A. Hierarki Prioritas Tangan (Arm Logic)
Sistem menggunakan logika prioritas untuk menghindari glitch animasi:
Zona Atas (Threshold Hidung): Jika koordinat Y Wrist < Nose, sistem memaksa state UP (Angkat Tangan), mengabaikan deteksi jari.
Zona Bawah: Jika tangan di bawah, sistem baru menghitung jumlah jari untuk pose spesifik (misal: 5 Jari, Peace).

B. Smoothing & Physics
Leaning (Kemiringan): Menggunakan rumus Current = Current + (Target - Current) * SmoothFactor. Teknik ini mencegah gerakan patah-patah (jitter) saat pengguna miring cepat.
Breathing (Nafas): Menggunakan rumus Y_Offset = Sin(Frame_Count * Speed) * Amplitude.

C. UI Rendering
Panel UI digambar manual menggunakan OpenCV dengan teknik Numpy Broadcasting untuk membuat warna secara efisien tanpa memperlambat FPS (Frame per Second).

| Tombol | Fungsi | Deskripsi |
| :---: | :--- | :--- |
| **`H`** | **Hide/Show Menu** | Menampilkan atau menyembunyikan panel bantuan navigasi. |
| **`ESC`** | **Keluar** | Menutup aplikasi. |
| **`P`** | **Global Transform** | Membuka editor posisi & skala karakter secara keseluruhan. |
| **`O`** | **Part Editor** | Membuka editor untuk kalibrasi per bagian (Mata/Mulut/Leher). |
| **`S`** | **Simpan** | Menyimpan konfigurasi posisi saat ini ke file JSON. |
| **`X`** | **Tutup Editor** | Menutup menu slider (P atau O). |
| **`K`** | **Sharpening** | (On/Off) Efek penajaman gambar agar karakter terlihat lebih detail. |
| **`L`** | **Leaning (Sway)** | (On/Off) Animasi miring tubuh mengikuti bahu pengguna. |
| **`Z`** | **Breathing** | (On/Off) Animasi bernafas otomatis (naik-turun). |
| **`.` / `,`** | **Ganti BG** | Mengganti latar belakang ke file berikutnya/sebelumnya. |

# Link Video Demo Vtuber: 
https://drive.google.com/file/d/1hIm1xg85SZnd_9NCEexyEhcE5rPbUQvd/view?usp=drive_link
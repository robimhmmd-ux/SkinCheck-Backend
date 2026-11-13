# Glosarium Aplikasi Laomera

Dokumen ini adalah sumber kebenaran **pengertian** istilah, fitur, cara pakai, dan teknologi aplikasi Laomera (dulu SkinCheck). Isinya **sama persis** dengan `APP_GLOSSARY` di `app/services/ai_assistant_service.py` — kalau ada yang berubah, ubah di kedua tempat.

Chatbot memakai glosarium ini untuk menjawab pertanyaan "apa itu ...?", "apa yang dimaksud ...?", "apa bedanya ...?", "gimana/bagaimana cara ...?", "apakah bisa ...?", dan "model/fitur ini untuk apa?" soal aplikasi. Pertanyaan soal **data** (jumlah, daftar, siapa yang paling sering) tetap dijawab dari database, bukan dari glosarium ini. Pertanyaan soal **kesehatan kulit** tetap dijawab dokumen RAG skincare, bukan dari glosarium ini.

## Status akun

### Akun yang ditangguhkan (suspended)

Akun yang ditangguhkan adalah akun yang dinonaktifkan sementara oleh admin — pemiliknya tidak bisa login dan tidak bisa memakai fitur apa pun sampai admin memulihkannya. Berbeda dengan akun pending (akun baru yang belum selesai verifikasi), akun ditangguhkan adalah akun yang sengaja diberhentikan sementara oleh admin.

### Akun pending (menunggu verifikasi OTP)

Akun pending adalah akun yang baru saja daftar tapi belum verifikasi nomor WhatsApp. Pemiliknya sudah bisa login, tapi semua fitur terkunci sampai dia memasukkan kode OTP 6 angka yang dikirim lewat WhatsApp, lalu melengkapi biodata.

### Akun menunggu biodata (awaiting_biodata)

Akun yang sudah lolos verifikasi kode OTP tapi belum melengkapi biodata — nama, umur (5–120 tahun), dan jenis kelamin. Setelah biodatanya diisi, akunnya jadi aktif dan semua fitur terbuka.

### Akun nonaktif (inactive)

Akun nonaktif adalah akun yang sedang tidak aktif dan pemiliknya tidak bisa login. Biasanya ini akun yang dinonaktifkan oleh admin — tombol hapus di panel Pengguna hanya menonaktifkan akun, bukan menghapusnya permanen, jadi admin bisa mengaktifkannya kembali.

### Akun aktif

Akun aktif adalah akun yang sudah menyelesaikan semua tahap verifikasi — kode OTP WhatsApp dan biodata — dan bisa memakai semua fitur aplikasi.

## Akun: yang bisa dan yang tidak bisa

### Ganti nama akun — BISA

Bisa — ganti nama akun lewat halaman Profil: ubah kolom nama, lalu tekan Simpan. Nama baru langsung berlaku dan tampil di dashboard serta sidebar. Yang tidak bisa diubah adalah email.

### Ganti email / Gmail — TIDAK BISA

Tidak bisa — email adalah identitas untuk masuk ke akun, jadi tidak bisa diubah setelah daftar. Di halaman Profil, kolom email memang dikunci. Kalau kamu perlu pakai email lain, daftar akun baru dengan email itu.

### Ganti password

Customer tidak bisa ganti password sendiri — tidak ada fitur ganti password di halaman Profil. Kalau lupa password, minta admin untuk mengatur ulang passwordmu lewat halaman Pengguna. Admin bisa ganti passwordnya sendiri lewat Profil Admin, dengan memasukkan password lama (password baru minimal 6 karakter).

### Admin menonaktifkan diri sendiri — TIDAK BISA

Tidak bisa — admin tidak bisa menonaktifkan akunnya sendiri, supaya website tidak sampai kehilangan adminnya. Kalau akun admin perlu dinonaktifkan, harus dilakukan oleh admin lain. Aturan yang sama berlaku untuk menghapus akun admin sendiri.

### Hapus akun

Customer tidak bisa menghapus akunnya sendiri — tidak ada tombol hapus akun di halaman Profil; kalau akunmu memang harus dihapus, minta admin untuk menonaktifkannya lewat halaman Pengguna. Admin menghapus pengguna lewat tombol hapus di halaman Pengguna — itu hanya menonaktifkan akun (status jadi nonaktif), bukan menghapus permanen.

## Verifikasi & OTP

### Verifikasi akun

Verifikasi akun ada tiga langkah: daftar (akun jadi pending), lalu masukkan kode OTP 6 angka yang dikirim ke WhatsApp (kode berlaku 10 menit dan sekali pakai), lalu isi biodata (nama, umur 5–120 tahun, jenis kelamin). Setelah itu akun jadi aktif dan semua fitur terbuka.

### OTP

OTP adalah kode 6 angka yang dikirim ke nomor WhatsApp kamu saat verifikasi akun. Kodenya berlaku 10 menit, sekali pakai, dan kalau mau minta ulang harus menunggu 60 detik. Kodenya tidak pernah ditampilkan di aplikasi — cuma kamu yang bisa melihatnya di WhatsApp.

## Cara pakai

### Cara chat dengan asisten

Buka tombol chat di pojok kanan bawah (ikon balon percakapan) — asisten Laomera siap menjawab soal kondisi kulitmu, produk, dan data akunmu. Riwayat chatmu tersimpan dan bisa dibuka lagi kapan pun; riwayat itu terhapus otomatis saat kamu logout.

### Admin menambah produk / pengguna

Admin menambah produk lewat halaman Produk (tombol 'Tambah Produk', isi merek, nama, kategori, deskripsi, kandungan, dan tautan Shopee/Tokopedia), dan menambah pengguna lewat halaman Pengguna (tombol 'Tambah Customer', isi nama, email, dan peran).

## Scan & skor

### Scan wajah

Scan adalah pemeriksaan kondisi kulit wajah lewat foto. Ada dua cara: foto tunggal (satu foto langsung dianalisis) dan multi-pose (kamera mengambil 12 foto selama 24 detik dari 6 arah, lalu hasilnya digabung). Caranya: buka halaman Scan Kulit dari sidebar, pilih foto tunggal atau multi-pose, lalu ikuti panduan kameranya sampai selesai.

### Scan multi-pose

Scan multi-pose adalah scan memakai kamera yang mengambil 12 foto selama 24 detik, dari 6 arah: tengah, kiri, kanan, atas, bawah, dan dekat. Setiap foto yang wajahnya terdeteksi dianalisis, lalu skor-skornya digabung jadi satu hasil.

### Skor 5 kondisi kulit

Ada 5 kondisi yang dinilai dari scan: jerawat, kulit kering, kulit berminyak, dark spots (noda hitam), dan kerutan. Setiap kondisi diberi skor 0–100% — ini skor KEPARAHAN: makin tinggi persentasenya, makin kuat kondisi itu terdeteksi di wajahmu.

### Skor kesehatan keseluruhan (overall score)

Skor kesehatan keseluruhan (overall score) adalah angka 0–100 yang merangkum hasil scanmu: 100 dikurangi rata-rata kelima skor kondisi. Makin tinggi makin sehat — di atas 75 berarti kulitmu prima, di bawah 35 berarti perlu perhatian ekstra.

## Teknologi aplikasi

### Model AI VGG16

Model AI yang menilai scanmu adalah VGG16 yang sudah dilatih khusus (file finetuned_best.h5) — model ini membaca foto wajah dan memberi skor 0–100 untuk 5 kondisi: jerawat, kulit kering, kulit berminyak, dark spots, dan kerutan. Model ini hanya menilai seberapa kuat ciri tiap kondisi terlihat di wajah — bukan mengukur luas atau lokasi jerawat.

### YuNet (deteksi wajah)

Sebelum dinilai, wajahmu dideteksi dulu oleh YuNet — kalau di foto tidak ada wajah yang terdeteksi, scan ditolak dan kamu diminta foto ulang. Jadi skor yang keluar selalu dihitung dari wajah, bukan dari latar foto.

### Modal (AI di cloud)

Modal adalah layanan cloud tempat model AI dijalankan — saat kamu scan, foto dikirim ke server Modal untuk dianalisis, jadi ponsel atau laptopmu tidak perlu menjalankan modelnya sendiri.

### TiDB (database)

TiDB adalah database yang dipakai aplikasi ini di server (database MySQL yang berjalan di cloud) — semua data akun, hasil scan, produk, dan pengaturan tersimpan di sana.

### Fonnte (OTP WhatsApp)

Fonnte adalah layanan yang mengirim kode OTP ke WhatsApp kamu saat verifikasi akun. Kalau kode OTP tidak sampai, biasanya masalahnya ada di pengaturan Fonnte (tokennya) — aplikasinya sendiri tetap jalan.

### Teknologi aplikasi (Next.js / FastAPI / Vercel)

Aplikasi ini dibangun dengan frontend Next.js dan backend FastAPI, dan di-hosting di Vercel — setiap kali ada perubahan kode yang di-push ke GitHub, websitenya langsung diperbarui otomatis.

## Fitur & halaman

### Marketplace

Marketplace adalah katalog produk skincare di aplikasi — kamu bisa cari produk berdasarkan kategori (face wash, toner, serum, moisturizer, sunscreen, mask, eksfoliator) dan membelinya lewat tautan Shopee atau Tokopedia di tiap produk. Tidak ada keranjang atau pembayaran di dalam aplikasi.

### Riwayat & Diagnosis

Riwayat & Diagnosis adalah arsip semua scan yang pernah kamu lakukan, lengkap dengan skor tiap kondisi dan perbandingan dengan scan sebelumnya — jadi kamu bisa melihat progres kesehatan kulitmu dari waktu ke waktu. Caranya: buka halaman Riwayat & Diagnosis dari sidebar — semua scanmu tampil dari yang terbaru. Panel 'Diagnosis Terkini' di atasnya menampilkan hasil scan terbarumu.

### Produk

Produk adalah item skincare di katalog aplikasi (marketplace) — masing-masing punya merek, nama, kategori, deskripsi, kandungan, dan tautan Shopee/Tokopedia. Admin mengelolanya di halaman Produk; rekomendasi produk ke customer hanya muncul kalau customer yang memintanya.

### AI Copilot

AI Copilot adalah halaman admin yang membaca tren masalah kulit pengguna dan mengusulkan produk apa yang sebaiknya ditambah ke katalog — misalnya kalau rata-rata jerawat pengguna tinggi tapi produk anti-jerawat di katalog masih sedikit.

### Obrolan Customer

Obrolan Customer adalah halaman admin untuk memantau pertanyaan yang diajukan customer ke chatbot — admin bisa melihat pertanyaan yang paling sering ditanyakan dan membaca riwayat chat tiap customer, supaya tahu fitur apa yang membingungkan dan update apa yang dibutuhkan.

### Tren Scan

Tren Scan adalah halaman admin yang menampilkan rata-rata skor 5 kondisi kulit dari semua pengguna, kondisi yang paling dominan, dan rekomendasi produk untuk tiap kondisi.

### Audit Scan

Audit Scan adalah halaman admin yang menampilkan semua pemindaian dari semua pengguna beserta status dan hasilnya — admin bisa memeriksa scan siapa saja, kapan, dan berhasil atau gagal.

### Log Aktivitas

Log Aktivitas adalah catatan semua kejadian penting di aplikasi — siapa mendaftar, siapa verifikasi, siapa scan — beserta waktunya. Admin melihatnya di halaman Log Aktivitas.

### Fitur aplikasi

Fitur aplikasi untuk customer: Dashboard (ringkasan kondisi kulit), Scan Kulit (foto tunggal atau multi-pose 12 foto), Riwayat & Diagnosis (arsip scan + perbandingan), Marketplace (katalog produk skincare), dan chatbot Laomera Asisten. Untuk admin, tambahannya: Pengguna, Produk, Audit Scan, AI Copilot, Obrolan Customer, Tren Scan, Konfigurasi, Log Aktivitas, dan Profil Admin.

### Laomera

Laomera (dulu bernama SkinCheck) adalah aplikasi web AI untuk memeriksa kesehatan kulit wajah — scan foto wajahmu, dapatkan skor 5 kondisi kulit, lalu dapatkan rekomendasi produk yang cocok.

## Peran

### Akun admin

Admin adalah pengelola aplikasi: dia bisa melihat dan mengatur pengguna, produk, audit scan, tren kulit, obrolan customer, konfigurasi, dan log aktivitas. Admin juga bisa menangguhkan atau menonaktifkan akun, dan mengubah status verifikasi pengguna.

### Customer

Customer adalah pengguna biasa aplikasi: dia bisa scan kulit, melihat dashboard dan riwayat diagnosisnya, belanja di marketplace, dan chat dengan asisten AI — tapi hanya soal akun dan kulitnya sendiri, bukan data pengguna lain.

## AI & pengaturan

### Provider AI (Gemini / NVIDIA)

Chatbot Laomera bisa memakai dua penyedia AI: Gemini (Google) atau NVIDIA (model Llama lewat API yang kompatibel dengan OpenAI). Admin memilihnya di Konfigurasi → tab AI, mengisi kunci API-nya, dan perubahannya langsung berlaku tanpa restart. Kalau kuncinya salah atau providernya mati, chatbot tetap menjawab dari jawaban lokal, bukan error.

### Konfigurasi admin

Konfigurasi adalah halaman pengaturan admin (menu Konfigurasi) dengan tiga tab: AI (provider Gemini/NVIDIA, kunci API, model, dan kuota), OTP (verifikasi WhatsApp dan aturan nomor), dan Admin (pengaturan akun admin). Perubahan di sana langsung berlaku tanpa restart.

---

## Ruang lingkup chatbot

Chatbot Laomera hanya menjawab tiga hal:

1. **Seluruh database aplikasi** — semua data untuk admin; data akun dan kulit sendiri untuk customer.
2. **Seluruh fitur dan teknologi aplikasi** — pengertian, cara pakai, yang bisa dan yang tidak bisa, dan teknologi di baliknya (glosarium ini).
3. **Kesehatan kulit wajah** — kondisi kulit, skor, perawatan, dan produk skincare (dijawab dokumen RAG skincare).

Di luar ketiga topik itu — politik, harga saham/crypto, pelajaran sekolah, masakan, olahraga, dan topik umum lainnya — chatbot menjawab jujur bahwa itu di luar yang bisa dibantu, **tanpa** menempelkan angka database dan **tanpa** dokumen yang tidak nyambung. Kalau pertanyaannya butuh tindakan medis (luka terbuka, obat keras, kondisi darurat), chatbot menyarankan dokter kulit spesialis.

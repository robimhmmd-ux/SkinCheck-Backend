# Menjalankan SkinCheck dengan TiDB Cloud (pengganti XAMPP MySQL)

TiDB Cloud kompatibel MySQL, jadi aplikasi ini tidak perlu ganti driver
atau dialek. Yang berubah hanya: host, port, database, dan TLS.

## 0. Keamanan dulu (wajib)

- JANGAN pernah mengirim password database di chat, dan JANGAN commit
  file `.env` atau file konfigurasi MCP yang berisi password ke GitHub.
- Kalau sebuah password sempat terkirim di chat / ter-commit, anggap
  bocor dan segera ganti di TiDB Cloud console.
- File `.gitignore` di project ini sudah mengabaikan `.env`.

## 1. Siapkan database di TiDB Cloud

1. Buat cluster TiDB Cloud Serverless (free tier).
2. Di console, buat database khusus untuk aplikasi ini, misal
   `skincheck_db`.
3. JANGAN pakai database sistem bawaan untuk tabel aplikasi.
4. Catat dari halaman Connect: host gateway, port `4000`, username
   (format `<prefix>.root`), dan password yang baru.

## 2. Isi `.env` (lokal saja, jangan di-commit)

```env
DB_CONNECTION=mysql
DB_HOST=<gateway dari TiDB Cloud>
DB_PORT=4000
DB_DATABASE=skincheck_db
DB_USERNAME=<prefix>.root
DB_PASSWORD=<password baru>
DB_SSL=true
DB_SSL_CA=
```

- `DB_SSL=true` wajib untuk endpoint publik TiDB Cloud (TLS).
- `DB_SSL_CA` boleh kosong: kalau `certifi` terinstall (sudah ada di
  `requirements.txt`), bundle CA-nya dipakai otomatis.
- Untuk MySQL lokal (XAMPP), kembalikan `DB_PORT=3306` dan
  `DB_SSL=false`.

## 3. Jalankan

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python setup_db.py           # bikin tabel + akun demo
python run.py                # buka http://127.0.0.1:8000
```

Catatan `setup_db.py`: kalau database sudah dibuat dari console,
langkah "CREATE DATABASE" di script boleh gagal dan itu tidak masalah
— script tetap lanjut membuat tabel.

## 4. MCP TiDB (opsional, bukan koneksi aplikasi)

Konfigurasi `mcpServers` / `tidb-mcp-server` (via `uvx`) itu hanya
untuk AI assistant (Cursor/Claude) supaya bisa query database
langsung. Aplikasi FastAPI TIDAK memakai MCP; aplikasi memakai `.env`
di atas. Simpan konfigurasi MCP di file lokal AI tool saja, jangan
di repo project.

## Perubahan kode yang sudah dilakukan (2026-10-28)

- `app/config/settings.py`: tambah `DB_SSL` dan `DB_SSL_CA`.
- `app/config/database.py`: tambah `mysql_connect_args()` — kalau
  `DB_SSL=true`, PyMySQL diberi `ssl_verify_cert`,
  `ssl_verify_identity`, dan `ssl_ca` (certifi kalau ada).
- `setup_db.py`: koneksi sementara pembuat database memakai TLS yang
  sama.
- `app/services/ai_service.py`: path model `.h5` relatif sekarang
  diselesaikan terhadap root project, jadi ketemu walau server
  dijalankan dari folder lain.
- `requirements.txt`: tambah `opencv-python`, `google-generativeai`,
  `bcrypt`, `certifi` yang sebelumnya hilang.
- `.gitignore` baru: mengabaikan `.env`, `__pycache__/`, `venv/`,
  file sqlite lokal, dan folder `storage/`/`uploads/`.

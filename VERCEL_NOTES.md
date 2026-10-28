# Backend di Vercel

Backend split ini (FastAPI API-only, `/api/v2`) bisa di-deploy ke Vercel
sebagai serverless function, dengan catatan yang sama seperti monolit:

- Entrypoint: `api/index.py` hanya me-re-export `app` dari `app.main`.
  `vercel.json` me-rewrite semua path ke `/api/index`.
- Dependensi function: `api/requirements.txt` (core saja, tanpa
  tensorflow / opencv / google-generativeai yang diimpor lazy).
- Database: **TiDB Cloud (MySQL-compatible)**, bukan SQLite dan bukan
  database lokal. Tabel sudah dibuat dengan `skincheck_setup.sql`.
  `setup_db.py` tidak berjalan otomatis di Vercel.
- Environment variables (isi di dashboard Vercel, JANGAN di-commit,
  JANGAN dikirim di chat): `SECRET_KEY`, `DB_CONNECTION=mysql`,
  `DB_HOST`, `DB_PORT=4000`, `DB_DATABASE=skincheck_db`,
  `DB_USERNAME`, `DB_PASSWORD`, `DB_SSL=true`,
  `FRONTEND_URL=https://skincheck-frontend-robimhmmd-uxs-projects.vercel.app`,
  `SESSION_SAME_SITE=none` (cross-site: frontend di vercel.app,
  backend di domain Vercel lain — cookie `SameSite=None` wajib `Secure`,
  dan kode sudah memaksanya otomatis saat nilai ini `none`).
- Upload scan: di Vercel filesystem read-only kecuali `/tmp`, jadi
  `image_service.py` sudah mengarahkan storage ke `/tmp/storage/scans`
  saat `VERCEL=1`.
- Batasan: analisis AI penuh (model `.h5` 174 MB, tidak ada di repo)
  tidak jalan di Vercel sampai inferensi dipindah ke layanan eksternal
  (mis. Modal). Yang jalan: auth, dashboard, history, products, dan
  scan upload dengan status `processing`.
- Setelah backend live: isi `NEXT_PUBLIC_API_URL` di Environment
  Variables project frontend Vercel dengan URL backend (tanpa trailing
  slash), lalu redeploy frontend, dan tes register → login →
  dashboard/history.

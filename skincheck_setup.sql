-- ============================================================================
-- SkinCheck — Database Setup (MySQL / TiDB Cloud)
-- ============================================================================
-- Sumber skema: model SQLAlchemy di repo SkinCheck-AI, commit 5d006e7
--   app/models/user.py, product.py, skin_scan.py, analysis_result.py,
--   activity_log.py, otp_token.py, system_setting.py, chat_message.py
--   (persis seperti yang dibuat `python setup_db.py` via Base.metadata.create_all)
--
-- Cara pakai (pilih salah satu):
--   A. TiDB Cloud console -> SQL Editor / Chat2Query -> paste seluruh file ini -> Run
--   B. MCP TiDB (di editor kamu) -> jalankan file ini sebagai query
--   C. MySQL CLI lokal:  mysql -h <host> -P 4000 -u <user> -p < skincheck_setup.sql
--      (untuk TiDB Cloud, koneksi harus TLS — lewat console / MCP lebih mudah)
--
-- File ini TIDAK berisi password, API key, atau kredensial apa pun.
-- Setelah database + tabel jadi, isi env var aplikasi (di dashboard Vercel
-- atau .env lokal) dengan: DB_CONNECTION=mysql, DB_HOST, DB_PORT=4000,
-- DB_DATABASE=skincheck_db, DB_USERNAME, DB_PASSWORD, DB_SSL=true
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Database
-- ----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS `skincheck_db`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `skincheck_db`;

-- ----------------------------------------------------------------------------
-- 2. Tabel (urutan disesuaikan dengan foreign key)
-- ----------------------------------------------------------------------------

-- Pengguna: customer & admin. password_hash diisi aplikasi saat register
-- (bcrypt), BUKAN password polos.
CREATE TABLE IF NOT EXISTS `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `email` VARCHAR(150) NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  `role` VARCHAR(50) DEFAULT 'customer',
  `status` VARCHAR(50) DEFAULT 'active',
  `profile_data` TEXT NULL,
  `created_at` DATETIME NULL,
  `updated_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_users_email` (`email`),
  INDEX `ix_users_email` (`email`)
);

-- Katalog produk skincare internal.
CREATE TABLE IF NOT EXISTS `products` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `brand` VARCHAR(100) NOT NULL,
  `name` VARCHAR(150) NOT NULL,
  `category` VARCHAR(100) NOT NULL,
  `description` TEXT NULL,
  `ingredients` TEXT NULL,
  `skin_concern` TEXT NULL,
  `usage_guidance` TEXT NULL,
  `shopee_link` TEXT NULL,
  `tokopedia_link` TEXT NULL,
  `status` VARCHAR(50) DEFAULT 'active',
  `created_at` DATETIME NULL,
  `updated_at` DATETIME NULL,
  PRIMARY KEY (`id`)
);

-- Pengaturan sistem key-value (nilai default aplikasi ada di kode;
-- yang di-seed di bawah hanya nilai non-rahasia).
CREATE TABLE IF NOT EXISTS `system_settings` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `key` VARCHAR(100) NOT NULL,
  `value` TEXT NULL,
  `description` VARCHAR(255) NULL,
  `updated_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_system_settings_key` (`key`),
  INDEX `ix_system_settings_key` (`key`)
);

-- Kode OTP 6 digit untuk verifikasi register & hapus akun.
CREATE TABLE IF NOT EXISTS `otp_tokens` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `email` VARCHAR(150) NOT NULL,
  `code` VARCHAR(10) NOT NULL,
  `purpose` VARCHAR(50) NOT NULL,
  `is_used` TINYINT(1) DEFAULT 0,
  `expires_at` DATETIME NOT NULL,
  `created_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  INDEX `ix_otp_tokens_email` (`email`)
);

-- Log aktivitas login/logout.
CREATE TABLE IF NOT EXISTS `activity_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `activity` VARCHAR(255) NOT NULL,
  `ip_address` VARCHAR(50) NULL,
  `created_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_activity_logs_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
);

-- Riwayat chat asisten per pengguna.
CREATE TABLE IF NOT EXISTS `chat_messages` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `role` VARCHAR(20) NOT NULL,
  `content` TEXT NOT NULL,
  `created_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `ix_chat_messages_user_id` (`user_id`),
  CONSTRAINT `fk_chat_messages_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
);

-- Sesi scan wajah (satu baris per foto yang di-upload).
CREATE TABLE IF NOT EXISTS `skin_scans` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `image_path` VARCHAR(255) NOT NULL,
  `status` VARCHAR(50) DEFAULT 'completed',
  `image_quality_score` FLOAT DEFAULT 1.0,
  `created_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_skin_scans_user_id`
    FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
);

-- Hasil analisis AI per scan: 5 kondisi utama + skor tambahan.
CREATE TABLE IF NOT EXISTS `analysis_results` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `skin_scan_id` INT NOT NULL,
  `acne_score` FLOAT NOT NULL DEFAULT 0.0,
  `dry_skin_score` FLOAT DEFAULT 0.0,
  `oily_skin_score` FLOAT DEFAULT 0.0,
  `dark_spots_score` FLOAT DEFAULT 0.0,
  `wrinkles_score` FLOAT DEFAULT 0.0,
  `redness_score` FLOAT DEFAULT 0.0,
  `texture_score` FLOAT DEFAULT 0.0,
  `hyperpigmentation_score` FLOAT DEFAULT 0.0,
  `overall_score` FLOAT NOT NULL DEFAULT 0.0,
  `ai_response` TEXT NULL,
  `created_at` DATETIME NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_analysis_results_skin_scan_id`
    FOREIGN KEY (`skin_scan_id`) REFERENCES `skin_scans` (`id`) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- 3. Seed: pengaturan sistem (hanya nilai non-rahasia)
--    GEMINI_API_KEY, MAIL_PASSWORD, dan WA_API_TOKEN sengaja TIDAK di-seed:
--    itu rahasia, isi lewat dashboard admin / env var, bukan lewat SQL.
-- ----------------------------------------------------------------------------
INSERT INTO `system_settings` (`key`, `value`, `description`) VALUES
  ('AI_MODEL', 'gemini-1.5-flash', 'Model LLM Aktif'),
  ('AI_MAX_TOKENS', '1024', 'Batasan Maksimal Output Token AI'),
  ('AI_TEMPERATURE', '0.7', 'Tingkat Kreativitas AI'),
  ('AI_DAILY_SCAN_LIMIT', '5', 'Batas Kuota Scan Wajah Harian'),
  ('AI_SIMULATION_MODE', '0', 'Mode Simulasi AI Fallback'),
  ('REGISTRATION_OTP_MODE', 'dual', 'Mode Alur Verifikasi OTP Registrasi'),
  ('MAINTENANCE_MODE', 'off', 'Status Mode Pemeliharaan Sistem'),
  ('VGG_THRESHOLD', '40', 'VGG16 Clinical Threshold %'),
  ('YOLO_CONFIDENCE', '0.25', 'YOLOv8 Acne Confidence Threshold')
ON DUPLICATE KEY UPDATE `value` = VALUES(`value`);

-- ----------------------------------------------------------------------------
-- 4. Seed: katalog produk (dari seed_products.py di repo)
--    skin_concern disimpan sebagai JSON string, sama seperti di kode.
-- ----------------------------------------------------------------------------
INSERT INTO `products`
  (`brand`, `name`, `category`, `description`, `ingredients`, `skin_concern`, `usage_guidance`, `status`) VALUES
  ('SkinCheck', 'Gentle Facial Cleanser', 'cleanser',
   'Pembersih wajah pH balanced bebas sulfat untuk kulit sensitif dan rentan berjerawat.',
   'Water, Glycerin, Sodium Cocoyl Isethionate, Centella Asiatica Extract, Tea Tree Leaf Oil',
   '["jerawat", "berminyak"]',
   'Gunakan pagi dan malam hari pada wajah basah, bilas hingga bersih.',
   'active'),
  ('SkinCheck', 'Niacinamide 5% Serum', 'serum',
   'Serum pencerah dan pelindung skin barrier dengan Niacinamide murni dan Zinc PCA.',
   'Niacinamide, Zinc PCA, Hyaluronic Acid, Centella Asiatica, Allantoin',
   '["kemerahan", "dark spots", "bekas jerawat"]',
   'Teteskan 2-3 tetes pada wajah setelah toner, ratakan perlahan.',
   'active'),
  ('SkinCheck', 'Daily Sunscreen SPF 50 PA++++', 'sunscreen',
   'Tabir surya ringan tanpa whitecast dengan perlindungan broad-spectrum UV.',
   'Zinc Oxide, Ethylhexyl Methoxycinnamate, Niacinamide, Tocopherol, Ceramide NP',
   '["semua", "anti-aging", "sun protection"]',
   'Aplikasikan sebanyak dua jari 15 menit sebelum terpapar sinar matahari.',
   'active'),
  ('SkinCheck', 'Hydrating Ceramide Moisturizer', 'moisturizer',
   'Pelembap restorasi barrier kulit dengan 5x Ceramides dan Hyaluronic Acid.',
   'Ceramide NP, Ceramide AP, Ceramide EOP, Hyaluronic Acid, Squalane, Shea Butter',
   '["kering", "kulit mengelupas", "wrinkles"]',
   'Oleskan secara merata pada wajah dan leher sebagai langkah akhir rutinitas.',
   'active'),
  ('SkinCheck', 'BHA Clarifying Acne Toner', 'toner',
   'Toner eksfoliasi lembut dengan Salicylic Acid 2% untuk membersihkan pori tersumbat.',
   'Salicylic Acid 2%, Tea Tree Extract, Witch Hazel, Willow Bark Extract',
   '["jerawat", "komedo", "pori-pori besar"]',
   'Gunakan 2-3 kali seminggu pada malam hari menggunakan kapas halus.',
   'active');

-- ----------------------------------------------------------------------------
-- 5. Akun pengguna — SENGAJA TIDAK DI-SEED DI SINI
-- ----------------------------------------------------------------------------
-- `setup_db.py` di repo memang membuat akun demo admin & customer, tapi
-- password demo itu hard-coded di script dan TIDAK AMAN untuk database
-- yang akan dipakai demo online (Vercel). Karena itu file ini tidak
-- memasukkan akun apa pun ke tabel `users`.
--
-- Cara membuat akun yang benar:
--   a. Daftar lewat halaman Register aplikasi (password langsung di-hash
--      bcrypt oleh aplikasi), lalu ubah role ke 'admin' kalau perlu:
--        UPDATE `users` SET `role` = 'admin' WHERE `email` = 'email-kamu';
--   b. Atau, hanya untuk tes lokal (bukan TiDB online), jalankan
--      `python setup_db.py` di laptop — itu yang membuat akun demo.
--
-- Contoh struktur INSERT (JANGAN dipakai apa adanya — ganti hash-nya
-- dengan hash bcrypt asli dari aplikasi, jangan pernah isi password polos):
--   INSERT INTO `users` (`name`, `email`, `password_hash`, `role`, `status`)
--   VALUES ('Nama Kamu', 'email-kamu@example.com', '<HASH_BCRYPT_DARI_APLIKASI>', 'admin', 'active');

-- ----------------------------------------------------------------------------
-- 6. Verifikasi (jalankan setelah file ini selesai di-run)
-- ----------------------------------------------------------------------------
-- SHOW TABLES;
-- SELECT COUNT(*) AS total_products FROM `products`;
-- SELECT `key`, `value` FROM `system_settings` ORDER BY `key`;

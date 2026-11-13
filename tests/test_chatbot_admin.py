"""Unit test otomatis jawaban chatbot ADMIN (Laomera).

Dibuat 2026-11-13 untuk perbaikan chatbot admin: total pengguna (bukan
customer saja), follow-up angka yang harus dijawab, fokus perawatan per
kondisi, ganti nama/email admin. Dijalankan pipeline CI di setiap push.

Tes ini memakai database sqlite SEMENTARA (tmp_path) — bukan database
live dan bukan file skincheck_db.sqlite repo — jadi CI hijau tanpa TiDB.
Semua angka/nama di sini adalah data seed tes, bukan data live.
"""

import sys
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "/home/hatch/workspace/projects/web-ai-skincheck/split/backend")

from app.config.database import Base  # noqa: E402
from app.models.analysis_result import AnalysisResult  # noqa: E402
from app.models.product import Product  # noqa: E402
from app.models.skin_scan import SkinScan  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.ai_assistant_service import (  # noqa: E402
    AIOrchestrator,
    ContextBuilder,
)


@pytest.fixture()
def db(tmp_path):
    """Seed minimal: 1 admin + 3 customer (2 aktif, 1 pending),
    3 scan completed, 2 produk aktif."""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine)
    s = S()

    def mk(name, email, status="active", role="customer"):
        u = User(name=name, email=email, password_hash="x", role=role,
                 status=status, created_at=datetime.utcnow())
        s.add(u)
        s.flush()
        return u

    admin = mk("Admin Mahraraja", "admin@laomera.com", role="admin")
    budi = mk("Budi", "budi@gmail.com")
    sari = mk("Sari", "sari@uji.id")
    mk("ijal", "rijal@gmail.com", status="pending")

    base = datetime.utcnow()
    for u, hours in [(budi, 6), (budi, 30), (sari, 3)]:
        s.add(SkinScan(user_id=u.id, image_path="/tmp/x.jpg",
                       status="completed", created_at=base - timedelta(hours=hours)))
    s.flush()
    # Budi dominan berminyak (80); Sari dominan jerawat (55).
    for scan, scores in [
        (s.query(SkinScan).filter(SkinScan.user_id == budi.id).order_by(SkinScan.created_at.desc()).first(),
         (25, 20, 80, 10, 5, 80)),
        (s.query(SkinScan).filter(SkinScan.user_id == sari.id).order_by(SkinScan.created_at.desc()).first(),
         (55, 20, 30, 15, 10, 70)),
    ]:
        ac, dr, oi, sp, wr, ov = scores
        s.add(AnalysisResult(skin_scan_id=scan.id, acne_score=ac, dry_skin_score=dr,
                             oily_skin_score=oi, dark_spots_score=sp,
                             wrinkles_score=wr, overall_score=ov))
    # Fokus profil Sari: kulit kering (label Indonesia).
    data = sari.get_profile_data() or {}
    data["skin_concerns"] = ["Kulit Kering"]
    sari.set_profile_data(data)
    for name in ["Serum Niacinamide", "Face Wash Gentle"]:
        s.add(Product(brand="Laomera", name=name, category="serum", status="active"))
    s.commit()
    yield s
    s.close()


@pytest.fixture()
def admin_id(db):
    return db.query(User).filter(User.role == "admin").first().id


def _ask(db, admin_id, q):
    return AIOrchestrator()._generate_local_response(
        q, db, [], user_id=admin_id, role="admin")


def test_dashboard_reports_total_users_not_customer_only(db, admin_id):
    """Kartu dashboard menampilkan TOTAL pengguna (4), bukan customer saja."""
    ans = _ask(db, admin_id, "baca angka dashboard")
    assert "Pengguna Terdaftar: 4 orang (3 aktif) — Customer 3, Admin 1" in ans
    assert "Total Customer: 3 orang (3 aktif)" not in ans
    assert "sama persis" not in ans
    assert "Volume Sesi Scan: 3 pemindaian" in ans
    assert "AI Success Rate: 100.0%" in ans
    assert "Katalog Formulasi: 2 produk (2 aktif di marketplace)" in ans
    assert "Distribusi 5 Pilar" in ans and "Jerawat 34%" in ans
    assert "Latensi Database" in ans
    assert "Menunggu verifikasi: 1 akun (ijal)" in ans


def test_dashboard_detail_explains_meaning(db, admin_id):
    ans = _ask(db, admin_id,
               "Baca angka-angka dashboard, sebutin angkanya, dan jelasin maksudnya.")
    assert "Artinya:" in ans
    assert "Rata-rata live" in ans
    assert "Success rate 100.0% mengukur" in ans


def test_followup_numbers_must_be_answered(db, admin_id):
    """'Sebutin angka pastinya satu per satu' HARUS dijawab, bukan ditolak."""
    ans = _ask(db, admin_id, "Sebutin angka pastinya satu per satu.")
    assert "Pengguna Terdaftar: 4 orang" in ans
    assert "di luar yang bisa aku bantu" not in ans


def test_all_users_listed(db, admin_id):
    ans = _ask(db, admin_id, "Sebutkan semua akun pengguna")
    for name in ["Admin Mahraraja", "Budi", "Sari", "ijal"]:
        assert name in ans
    assert "(4 orang)" in ans


def test_care_focus_per_condition(db, admin_id):
    ans = _ask(db, admin_id,
               "Akun yang menunjukkan fokus perawatannya jerawat siapa aja, "
               "lalu berminyak siapa aja, dan 5 output itu tanyain satu-satu")
    assert "1. Jerawat" in ans and "Sari (55%)" in ans
    assert "2. Kulit kering" in ans and "Sari (20%)" in ans
    assert "3. Kulit berminyak" in ans and "Budi (80%)" in ans
    assert "4. Dark spots" in ans and "5. Kerutan" in ans
    assert "Belum ada data fokus perawatan: ijal" in ans


def test_rename_admin_howto(db, admin_id):
    ans = _ask(db, admin_id, "Cara ganti nama admin gimana")
    assert "Profil Admin" in ans
    assert "PATCH /api/v2/admin/profile" in ans


def test_change_admin_email_rejected(db, admin_id):
    ans = _ask(db, admin_id, "Bisa ga ganti gmail admin")
    assert "Tidak bisa" in ans
    assert "Email tidak dapat diubah" in ans


def test_admin_context_snapshot_totals(db, admin_id):
    """Konteks LLM §1: total pengguna, bukan customer-only."""
    admin = db.query(User).filter(User.id == admin_id).first()
    ctx = ContextBuilder.build_admin_database_context(
        db, "", admin_user={"id": admin.id, "name": admin.name, "email": admin.email})
    assert "Pengguna Terdaftar: 4 orang (3 aktif)" in ctx
    assert "Customer: 3, Admin: 1" in ctx
    assert "Total Customer: 3 orang (3 Aktif)" not in ctx
    assert "BUKAN kartu dashboard" in ctx

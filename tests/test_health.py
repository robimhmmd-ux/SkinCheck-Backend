"""Unit test otomatis backend SkinCheck (Laomera).

Dibuat 2026-11-13 untuk rubrik RPL (Testing & GitHub DevOps/CI-CD):
pipeline CI menjalankan pytest ini di setiap push ke main.

Tes ini sengaja TIDAK menyentuh database — endpoint yang diuji adalah
endpoint publik tanpa akses DB, supaya CI hijau tanpa perlu TiDB.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    """GET /api/v2/health harus 200 (tanpa DB)."""
    r = client.get("/api/v2/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "status" in data or "ok" in data or "ai_status" in data


def test_me_logged_out_unauthorized():
    """GET /api/v2/me tanpa sesi harus 401 (bukan 500)."""
    r = client.get("/api/v2/me")
    assert r.status_code == 401

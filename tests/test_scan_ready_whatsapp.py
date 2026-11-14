"""Unit test notifikasi WA 'hasil scan siap' (Laomera).

Dibuat 2026-11-14: setelah analisis scan selesai sukses, customer login
yang punya nomor HP di profile_data["phone"] otomatis menerima WA via
Fonnte — tanpa konfigurasi apa pun. Tes memakai database sqlite SEMENTARA
(tmp_path) + TestClient dengan dependency get_db di-override, dan
_send_whatsapp di-mock (tidak ada pengiriman WA sungguhan, tidak ada
kunci Fonnte asli).

Kasus yang diuji:
- WA terkirim saat scan selesai sukses (single + multi upload).
- WA TIDAK terkirim untuk tamu/tester, customer tanpa nomor HP, dan
  scan yang gagal dianalisis.
- Kegagalan kirim WA (mis. token habis / Fonnte down) TIDAK menggagalkan
  scan — respons tetap 201 + status completed.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "/home/hatch/workspace/projects/web-ai-skincheck/split/backend")

from fastapi.testclient import TestClient  # noqa: E402

from app.config.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.fonnte_otp import FonnteSendError  # noqa: E402
from app.services.image_service import ImageService  # noqa: E402

_AI_DATA = {
    "acne_score": 18.5,
    "dry_skin_score": 16.7,
    "oily_skin_score": 31.1,
    "dark_spots_score": 72.9,
    "wrinkles_score": 47.1,
    "redness_score": 0.0,
    "texture_score": 0.0,
    "hyperpigmentation_score": 0.0,
    "overall_score": 62.7,
    "confidence": 0.9,
    "frames_analyzed": 1,
    "analysis_status": "ok",
}

_AI_DATA_MULTI = dict(
    _AI_DATA,
    frames_passed_yunet=2,
    frames_used_vgg=2,
    per_pose={},
    statements=[],
)


class _FakeProvider:
    """Pengganti KerasComputerVisionProvider — tanpa model .h5 / TensorFlow."""

    def __init__(self, fail: bool = False):
        self._fail = fail

    def analyze_skin_condition(self, path):
        if self._fail:
            raise RuntimeError("model tidak tersedia")
        return dict(_AI_DATA)

    def analyze_multi_frame_skin_condition(self, paths, poses):
        if self._fail:
            raise RuntimeError("model tidak tersedia")
        return dict(_AI_DATA_MULTI)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """DB sqlite + TestClient + mock AI/ImageService/_send_whatsapp.

    Mengembalikan (client, session, budi_id, sari_id, wa_mock).
    """
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine)
    s = S()

    budi = User(name="Budi", email="budi@gmail.com", password_hash="x",
                role="customer", status="active", created_at=datetime.utcnow())
    budi.set_profile_data({"phone": "6281234567890"})
    s.add(budi)
    sari = User(name="Sari", email="sari@uji.id", password_hash="x",
                role="customer", status="active", created_at=datetime.utcnow())
    s.add(sari)
    s.commit()
    budi_id, sari_id = budi.id, sari.id

    def _override_get_db():
        yield s

    app.dependency_overrides[get_db] = _override_get_db

    async def _fake_process(file):
        return "test.jpg"

    monkeypatch.setattr(ImageService, "process_and_save_scan",
                        staticmethod(_fake_process))
    monkeypatch.setattr(ImageService, "get_image_path",
                        staticmethod(lambda filename: Path("/tmp/test.jpg")))
    monkeypatch.setattr("app.services.ai_service.KerasComputerVisionProvider",
                        _FakeProvider)
    wa_mock = Mock()
    monkeypatch.setattr("app.routes.json_api._send_whatsapp", wa_mock)

    client = TestClient(app)
    yield client, s, budi_id, sari_id, wa_mock
    app.dependency_overrides.clear()
    s.close()


def _as_user(monkeypatch, user_id, name, email, is_guest=False):
    """Ganti _require_verified_customer dengan user palsu untuk satu tes."""
    user = {"id": user_id, "name": name, "email": email, "role": "customer"}
    if is_guest:
        user["is_guest"] = True
    monkeypatch.setattr(
        "app.routes.json_api._require_verified_customer",
        lambda request, db, allow_guest=False: user,
    )


def test_single_upload_sends_wa_to_customer_with_phone(env, monkeypatch):
    client, _s, budi_id, _sari_id, wa_mock = env
    _as_user(monkeypatch, budi_id, "Budi", "budi@gmail.com")

    r = client.post(
        "/api/v2/scan/upload",
        files=[("file", ("wajah.jpg", b"fake-image-bytes", "image/jpeg"))],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "completed"
    assert r.json()["analysis"]["overall_score"] == 62.7

    wa_mock.assert_called_once()
    args, _kwargs = wa_mock.call_args
    assert args[0] == "6281234567890"
    assert "Halo Budi!" in args[1]
    assert "62.7/100" in args[1]


def test_multi_upload_sends_wa(env, monkeypatch):
    client, _s, budi_id, _sari_id, wa_mock = env
    _as_user(monkeypatch, budi_id, "Budi", "budi@gmail.com")

    r = client.post(
        "/api/v2/scan/multi-upload",
        files=[
            ("files", ("a.jpg", b"fake-image-bytes", "image/jpeg")),
            ("files", ("b.jpg", b"fake-image-bytes", "image/jpeg")),
        ],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "completed"

    wa_mock.assert_called_once()
    args, _kwargs = wa_mock.call_args
    assert args[0] == "6281234567890"
    assert "Halo Budi!" in args[1]


def test_guest_scan_sends_no_wa(env, monkeypatch):
    client, _s, _budi_id, _sari_id, wa_mock = env
    _as_user(monkeypatch, 1, "Tester", "tester@uji.id", is_guest=True)

    r = client.post(
        "/api/v2/scan/upload",
        files=[("file", ("wajah.jpg", b"fake-image-bytes", "image/jpeg"))],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "completed"
    wa_mock.assert_not_called()


def test_customer_without_phone_sends_no_wa(env, monkeypatch):
    client, _s, _budi_id, sari_id, wa_mock = env
    _as_user(monkeypatch, sari_id, "Sari", "sari@uji.id")

    r = client.post(
        "/api/v2/scan/upload",
        files=[("file", ("wajah.jpg", b"fake-image-bytes", "image/jpeg"))],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "completed"
    wa_mock.assert_not_called()


def test_failed_scan_sends_no_wa(env, monkeypatch):
    client, _s, budi_id, _sari_id, wa_mock = env
    _as_user(monkeypatch, budi_id, "Budi", "budi@gmail.com")
    monkeypatch.setattr(
        "app.services.ai_service.KerasComputerVisionProvider",
        lambda: _FakeProvider(fail=True),
    )

    r = client.post(
        "/api/v2/scan/upload",
        files=[("file", ("wajah.jpg", b"fake-image-bytes", "image/jpeg"))],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "failed"
    wa_mock.assert_not_called()


def test_wa_failure_does_not_break_scan(env, monkeypatch):
    client, _s, budi_id, _sari_id, wa_mock = env
    _as_user(monkeypatch, budi_id, "Budi", "budi@gmail.com")
    wa_mock.side_effect = FonnteSendError("Gagal menghubungi Fonnte.")

    r = client.post(
        "/api/v2/scan/upload",
        files=[("file", ("wajah.jpg", b"fake-image-bytes", "image/jpeg"))],
    )
    assert r.status_code == 201
    assert r.json()["status"] == "completed"
    assert r.json()["analysis"]["overall_score"] == 62.7
    wa_mock.assert_called_once()

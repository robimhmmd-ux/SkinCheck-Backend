"""Unit test verifikasi provider AI di /admin/settings (Laomera).

Dibuat 2026-11-14 untuk 3 permintaan roboy:
1. Daftar model Gemini di modal tambah-key hanya model GRATIS yang
   benar-benar jalan + model yang sedang terkonfigurasi (1.5/2.0/2.5 mati).
2. Tombol provider hanya muncul bila kunci tersimpan DAN tes terakhir
   lolos — status terverifikasi persisten di SystemSetting
   AI_VERIFIED_PROVIDERS (JSON array). test-ai sukses -> tandai, gagal ->
   hapus tanda (+ provider aktif yang gagal jatuh ke terverifikasi lain);
   kunci/model diubah via PATCH -> hapus tanda (uji ulang).
3. GET /settings mengembalikan preview kunci (4 awal + ... + 4 akhir),
   TIDAK PERNAH kunci lengkap.

Tes memakai database sqlite SEMENTARA (tmp_path) + TestClient dengan
dependency get_db di-override; _require_admin dan _require_chat_user
di-mock; tidak ada panggilan jaringan sungguhan (httpx.get dan
google.generativeai di-mock), tidak ada kunci asli.
"""

import sys
import types
from datetime import datetime
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "/home/hatch/workspace/projects/web-ai-skincheck/split/backend")

from fastapi.testclient import TestClient  # noqa: E402

from app.config.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.system_setting import SystemSetting  # noqa: E402
from app.models.user import User  # noqa: E402
from app.routes.admin_api import _key_preview  # noqa: E402

_GEMINI_KEY = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234"
_GROQ_KEY = "gsk_abcdefgh12345678"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """DB sqlite + TestClient + admin palsu. Mengembalikan (client, session)."""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine)
    s = S()

    admin = User(name="Admin Mahraraja", email="admin@laomera.com",
                 password_hash="x", role="admin", status="active",
                 created_at=datetime.utcnow())
    s.add(admin)
    s.commit()

    def _override_get_db():
        yield s

    app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr(
        "app.routes.admin_api._require_admin",
        lambda request: {"id": admin.id, "name": "Admin Mahraraja",
                         "email": "admin@laomera.com", "role": "admin"},
    )
    monkeypatch.setattr(
        "app.routes.json_api._require_chat_user",
        lambda request, db: {"id": admin.id, "name": "Admin Mahraraja",
                             "email": "admin@laomera.com", "role": "admin"},
    )

    client = TestClient(app)
    yield client, s
    app.dependency_overrides.clear()
    s.close()


def _settings_map(client):
    r = client.get("/api/v2/admin/settings")
    assert r.status_code == 200
    return {s["key"]: s for s in r.json()["settings"]}


def _verified(client):
    import json
    raw = _settings_map(client)["AI_VERIFIED_PROVIDERS"]["value"]
    return json.loads(raw)


def _set(db, key, value):
    SystemSetting.set_value(db, key, value)


# ---------------------------------------------------------------------------
# 1. Filter daftar model Gemini
# ---------------------------------------------------------------------------

def _fake_gemini_models(monkeypatch):
    """httpx.get palsu: /v1beta/models mengembalikan model mati + hidup."""
    models = []
    for name in [
        "gemini-1.5-flash", "gemini-1.5-pro",       # mati (1.5 pensiun)
        "gemini-2.0-flash",                          # mati (2.0)
        "gemini-2.5-flash", "gemini-2.5-pro",        # tidak jalan kunci baru
        "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash",
        "gemini-3.1-flash-lite", "gemini-3.5-flash",  # allowlist gratis
    ]:
        models.append({"name": f"models/{name}",
                       "supportedGenerationMethods": ["generateContent"]})
    models.append({"name": "models/gemini-embedding-001",
                   "supportedGenerationMethods": ["embedContent"]})

    class _Resp:
        status_code = 200

        def json(self):
            return {"models": models}

    def _fake_get(url, **kwargs):
        return _Resp()

    monkeypatch.setattr("httpx.get", _fake_get)


def test_detect_gemini_filters_dead_models(env, monkeypatch):
    client, s = env
    _set(s, "AI_MODEL", "gemini-3.5-flash-lite")
    _fake_gemini_models(monkeypatch)

    r = client.post("/api/v2/admin/settings/detect-provider",
                    json={"api_key": "AIzaFakeKey1234567890"})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "gemini"
    models = body["models"]
    # Model mati TIDAK muncul.
    for dead in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash",
                 "gemini-2.5-flash", "gemini-2.5-pro", "gemini-embedding-001"]:
        assert dead not in models, f"{dead} seharusnya disaring"
    # Model allowlist muncul, dalam urutan allowlist.
    assert models == [
        "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash",
        "gemini-3.1-flash-lite", "gemini-3.5-flash",
    ]


def test_detect_gemini_keeps_configured_model_outside_allowlist(env, monkeypatch):
    client, s = env
    _set(s, "AI_MODEL", "gemini-custom-lama")
    _fake_gemini_models(monkeypatch)

    r = client.post("/api/v2/admin/settings/detect-provider",
                    json={"api_key": "AIzaFakeKey1234567890"})
    assert r.status_code == 200
    models = r.json()["models"]
    # Model terkonfigurasi tetap muncul (ditempel di akhir).
    assert models[-1] == "gemini-custom-lama"
    assert "gemini-1.5-flash" not in models


# ---------------------------------------------------------------------------
# 2. Flag terverifikasi: test-ai sukses / gagal / ganti kunci
# ---------------------------------------------------------------------------

def _fake_genai(monkeypatch, fail=False):
    """Suntik modul google.generativeai palsu ke sys.modules."""
    fake = types.ModuleType("google.generativeai")
    fake.configure = lambda api_key: None

    class _FakeModel:
        def __init__(self, name):
            self.name = name

        def generate_content(self, prompt, request_options=None):
            if fail:
                raise Exception("401 API key not valid")
            return types.SimpleNamespace(text="Koneksi AI SkinCheck Berhasil")

    fake.GenerativeModel = _FakeModel
    parent = types.ModuleType("google")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "google", parent)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)


def test_test_ai_success_marks_gemini_verified(env, monkeypatch):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "AI_MODEL", "gemini-3.5-flash-lite")
    _fake_genai(monkeypatch)

    r = client.post("/api/v2/admin/settings/test-ai", json={"provider": "gemini"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["provider"] == "gemini"
    assert "gemini" in _verified(client)


def test_test_ai_failure_removes_verified(env, monkeypatch):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "AI_MODEL", "gemini-3.5-flash-lite")
    _set(s, "AI_VERIFIED_PROVIDERS", '["gemini", "groq"]')
    _fake_genai(monkeypatch, fail=True)

    r = client.post("/api/v2/admin/settings/test-ai", json={"provider": "gemini"})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert _verified(client) == ["groq"]


def test_test_ai_success_marks_groq_verified(env, monkeypatch):
    client, s = env
    _set(s, "GROQ_API_KEY", _GROQ_KEY)
    _set(s, "AI_MODEL_GROQ", "llama-3.3-70b-versatile")
    monkeypatch.setattr(
        "app.services.ai_assistant_service._call_openai_compatible_chat",
        lambda db, provider, prompt: "Koneksi AI SkinCheck Berhasil",
    )

    r = client.post("/api/v2/admin/settings/test-ai", json={"provider": "groq"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "groq" in _verified(client)


def test_test_ai_failure_active_provider_falls_back_to_verified(env, monkeypatch):
    client, s = env
    _set(s, "AI_PROVIDER", "groq")
    _set(s, "AI_VERIFIED_PROVIDERS", '["gemini", "groq"]')
    _set(s, "GROQ_API_KEY", _GROQ_KEY)
    monkeypatch.setattr(
        "app.services.ai_assistant_service._call_openai_compatible_chat",
        lambda db, provider, prompt: None,  # Groq tidak menjawab
    )

    r = client.post("/api/v2/admin/settings/test-ai", json={"provider": "groq"})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert _verified(client) == ["gemini"]
    # Provider aktif jatuh ke Gemini (satu-satunya yang terverifikasi).
    assert _settings_map(client)["AI_PROVIDER"]["value"] == "gemini"


def test_test_ai_without_provider_tests_active(env, monkeypatch):
    """Tanpa body provider: yang diuji provider aktif (kompatibel lama)."""
    client, s = env
    _set(s, "AI_PROVIDER", "gemini")
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _fake_genai(monkeypatch)

    r = client.post("/api/v2/admin/settings/test-ai", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["provider"] == "gemini"
    assert "gemini" in _verified(client)


def test_patch_key_change_invalidates_verified(env):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "AI_MODEL", "gemini-3.5-flash-lite")
    _set(s, "AI_VERIFIED_PROVIDERS", '["gemini"]')

    # Simpan kunci yang SAMA -> tanda tetap.
    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"GEMINI_API_KEY": _GEMINI_KEY}})
    assert r.status_code == 200
    assert _verified(client) == ["gemini"]

    # Ganti kunci -> tanda hilang (harus uji ulang).
    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"GEMINI_API_KEY": "AIzaSyNewKeyBaru1234567890"}})
    assert r.status_code == 200
    assert _verified(client) == []


def test_patch_model_change_invalidates_verified(env):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "AI_MODEL", "gemini-3.5-flash-lite")
    _set(s, "AI_VERIFIED_PROVIDERS", '["gemini"]')

    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"AI_MODEL": "gemini-3.6-flash"}})
    assert r.status_code == 200
    assert _verified(client) == []


def test_patch_unrelated_setting_keeps_verified(env):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "AI_VERIFIED_PROVIDERS", '["gemini"]')

    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"AI_MAX_TOKENS": "512"}})
    assert r.status_code == 200
    assert _verified(client) == ["gemini"]


def test_patch_verified_providers_rejects_invalid(env):
    client, s = env
    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"AI_VERIFIED_PROVIDERS": "bukan-json"}})
    assert r.status_code == 400
    r = client.patch("/api/v2/admin/settings",
                     json={"settings": {"AI_VERIFIED_PROVIDERS": '["deepseek"]'}})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 3. Preview kunci
# ---------------------------------------------------------------------------

def test_key_preview_format():
    assert _key_preview(_GEMINI_KEY) == "AIza...1234"
    assert _key_preview(_GROQ_KEY) == "gsk_...5678"
    assert _key_preview("") == ""
    assert _key_preview(None) == ""
    assert _key_preview("pendek") == ""  # <= 8 karakter: tidak di-preview


def test_settings_returns_key_preview_not_full_key(env):
    client, s = env
    _set(s, "GEMINI_API_KEY", _GEMINI_KEY)
    _set(s, "GROQ_API_KEY", _GROQ_KEY)

    r = client.get("/api/v2/admin/settings")
    assert r.status_code == 200
    body_text = r.text
    # Kunci LENGKAP tidak pernah muncul di respons.
    assert _GEMINI_KEY not in body_text
    assert _GROQ_KEY not in body_text
    m = _settings_map(client)
    assert m["GEMINI_API_KEY"]["preview"] == "AIza...1234"
    assert m["GEMINI_API_KEY"]["configured"] is True
    assert m["GROQ_API_KEY"]["preview"] == "gsk_...5678"
    assert m["NVIDIA_API_KEY"]["preview"] == ""
    assert m["NVIDIA_API_KEY"]["configured"] is False


# ---------------------------------------------------------------------------
# 4. Chatbot fallback lokal saat tidak ada provider terverifikasi
# ---------------------------------------------------------------------------

def test_chat_falls_back_locally_when_nothing_verified(env):
    client, s = env
    _set(s, "AI_VERIFIED_PROVIDERS", "[]")
    _set(s, "AI_PROVIDER", "gemini")
    # Tidak ada kunci AI sama sekali.

    r = client.post("/api/v2/assistant/chat", json={"message": "halo"})
    assert r.status_code == 200  # TIDAK 500
    data = r.json()
    assert data["status"] == "success"
    assert isinstance(data["response"], str) and len(data["response"]) > 0

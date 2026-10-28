# SkinCheck — Backend (API-only, Python / FastAPI)

This is the **backend half** of the split SkinCheck project. It is a
JSON-only FastAPI API for the separate Next.js frontend repo. It is
split from the monolith at commit `5d006e7` of
`robimhmmd-ux/SkinCheck-AI` (the monolith is read-only for this split;
this backend was copied from it, not edited in place).

What changed in the split:
- **Removed:** Jinja templates (`app/views/`), static assets
  (`app/static/`), HTML routes. Static files and pages are now the
  frontend repo's job — this backend serves no HTML and no `/static`.
- **Kept:** SQLAlchemy models, DB config (MySQL/TiDB with TLS),
  auth utils (bcrypt session auth), services, `setup_db.py`,
  `seed_products.py`, and `skincheck_setup.sql`.
- **Added:** `app/routes/json_api.py` — the `/api/v2` JSON API.
- **Not mounted:** the legacy HTML routers/controllers are still on
  disk under `app/routes/` / `app/controllers/` as reference, but
  `app/routes/__init__.py` does not import them and `app/main.py`
  does not mount them, because they render templates that no longer
  exist here. Do not re-mount them without restoring the template layer.

## API v2 (all JSON, never TemplateResponse)

Base: the API root is the FastAPI app itself; v2 lives under `/api/v2`.
Legacy `/api/v1/health` is also kept.

| Method & path | Auth | What it does |
|---|---|---|
| `GET /api/v2/health` | public | Mirrors `/api/v1/health`. No DB access. |
| `POST /api/v2/auth/login` | public | `{email, password}` → `{user:{id,name,email,role}}` or 401. Sets `request.session["user"]` exactly like the monolith's `AuthController.handle_customer_login`, and logs an `ActivityLog` row. |
| `POST /api/v2/auth/register` | public | `{name, email, password, skin_concerns?}` → 201 `{user}` + session. **Direct registration (OTP bypass).** The monolith's OTP email flow stays in the monolith for now; this endpoint mirrors its `REGISTRATION_OTP_MODE=bypass` branch. Skin concerns are stored in `users.profile_data` JSON (the `User` model has no `phone`/`skin_concerns` columns). Duplicate email → 409. Password < 6 chars → 400. |
| `POST /api/v2/auth/logout` | session | Clears the session. |
| `GET /api/v2/me` | session or 401 | Current session user. No DB access. |
| `GET /api/v2/dashboard` | customer | Latest scan + last 10 scans with analysis scores. |
| `GET /api/v2/history` | customer | Own scans, most recent first (`?limit=`, default 50, max 100), each with its analysis summary or `null`. |
| `GET /api/v2/history/{id}` | customer | One own scan + analysis, or 404 (404 also covers "belongs to another user", so ownership is not leaked). |
| `POST /api/v2/scan/upload` | customer, multipart `file` | Validates + stores the image via `ImageService.process_and_save_scan`, creates a `SkinScan` row with `status="processing"`, returns `{scan_id, status}`. **No AI analysis is run here** and `ai_service` is never imported, so TensorFlow / the `.h5` model are not required. A follow-up endpoint must pick up `processing` rows and run the model. |
| `GET /api/v2/products` | public | Active products list. DB down → 503 JSON, not 500 HTML. |
| `GET /api/v2/admin/summary` | admin | Counts: `users`, `scans`, `products`, `active_products`. Non-admin → 403, logged out → 401. |

DB-dependent endpoints return `503 {"detail": "Database unavailable"}`
as JSON when the database cannot be reached, instead of a 500 page.

## Auth & CORS

- Auth is the monolith's session-cookie auth: `SessionMiddleware`,
  cookie `skincheck_session`, 7 days, `same_site="lax"`.
- CORS: origins from env `FRONTEND_URL` (comma-separated, default
  `http://localhost:3000`), `allow_credentials=True`, all methods
  and headers allowed.
- **Caveat:** `same_site="lax"` works when frontend and API share a
  site (same host, or Next.js rewrites/proxy to the API). For a truly
  cross-site production setup (frontend on one domain, API on another),
  the cookie will need `same_site="none"` + `https_only=True` — change
  it in `app/main.py` only when that deployment shape is confirmed.

## Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in .env (see below) — never commit .env
python setup_db.py     # creates the database + tables (+ demo accounts)
python run.py          # http://127.0.0.1:8000
# or: uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`setup_db.py` also works against an already-created database: if the
database already exists (e.g. created in the TiDB Cloud console), its
`CREATE DATABASE` step may fail harmlessly and table creation still
runs. The same schema is also available as plain SQL in
`skincheck_setup.sql` (database + 8 tables + product/system-setting
seeds; demo accounts are deliberately **not** in that SQL file — see
its header comment).

To seed the product catalogue only: `python seed_products.py`.

## .env variables

| Variable | Required? | Notes |
|---|---|---|
| `SECRET_KEY` | yes | Long random string. Never use the code's insecure default. |
| `DB_CONNECTION` | yes | `mysql` |
| `DB_HOST` | yes | Database host (TiDB gateway or `127.0.0.1` locally) |
| `DB_PORT` | yes | `4000` for TiDB Cloud, `3306` for local MySQL/XAMPP |
| `DB_DATABASE` | yes | `skincheck_db` — the dedicated app database, not a system database |
| `DB_USERNAME` | yes | Database user |
| `DB_PASSWORD` | yes | Database password |
| `DB_SSL` | yes for TiDB | `true` for TiDB Cloud (public endpoint requires TLS), `false` locally |
| `DB_SSL_CA` | no | CA bundle path; empty = certifi's bundle if installed |
| `FRONTEND_URL` | no | CORS origin(s), default `http://localhost:3000` |
| `GEMINI_API_KEY` | optional | Needed only for assistant/AI endpoints |
| `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_FROM_NAME` | optional | Needed only for OTP email |
| `LLM_PROVIDER` | no | Default `gemini` |
| `SKINCHECK_STORAGE_DIR` | no | Override for scan uploads; otherwise `storage/scans`, or `/tmp/storage/scans` when `VERCEL=1` |

WhatsApp (Fonnte) tokens are **not** `.env` variables: they live in
`system_settings` via the admin dashboard, as in the monolith.
`VECTOR_DB_URL` appears in the old `.env.example` but is never read by
the code, so it is not listed here.

## TiDB notes

TiDB Cloud is MySQL-compatible, so no driver/dialect change was needed.
See `TIDB_SETUP.md` (carried over from the monolith) for the console
steps. Key points: create a dedicated `skincheck_db` in the console,
use port `4000`, set `DB_SSL=true`, and never commit `.env` or any
file containing a password.

## Dependencies & the model

`requirements.txt` pins `fastapi==0.115.0`, `starlette==0.38.6`,
`jinja2==3.1.4` (jinja is not used by this API-only backend — it is
pinned only so a legacy import cannot silently pull a breaking
version, and it can be deleted once the legacy controllers are gone).

`tensorflow`, `opencv-python`, and `google-generativeai` are **optional
and lazy**: they are imported only inside service functions, never at
app import time. The API imports and serves auth, products, history,
and scan upload without them. The `.h5` model (174 MB) is not in this
repo and is not needed to run the API; it is only needed by a future
AI-analysis endpoint.

## Frontend repo

The frontend is a **separate repo** (Next.js). It is not in this
folder. The frontend calls this backend's `/api/v2` endpoints with
`credentials: "include"` (so the session cookie is sent) and the
backend's `FRONTEND_URL` must list the frontend's origin.

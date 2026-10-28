"""SkinCheck backend — API-only FastAPI app (split from the monolith).

Differences from the monolith's ``app/main.py``:
- No Jinja templates, no ``/static`` mount. Static assets and pages
  are the Next.js frontend repo's job.
- Only JSON routers are mounted: /api/v2 (new) and /api/v1/health
  (kept from the monolith).
- CORS is enabled for the frontend origin(s) in ``FRONTEND_URL``
  (comma-separated, default http://localhost:3000), with credentials
  allowed so the session cookie can be sent by the frontend.
- Auth failures return JSON (401), not HTML redirects. The monolith's
  ``AuthException`` redirect handler is therefore replaced with a JSON
  handler below.

Session cookie caveat (kept as in the monolith, per the split task):
- SessionMiddleware uses ``same_site="lax"`` and
  ``https_only=not APP_DEBUG``. That works when the frontend and the
  API share a site (e.g. Next.js rewrites / proxying to the API, or
  same-host development). For a truly cross-site production setup
  (frontend on vercel.app, API on another domain), the cookie will
  need ``same_site="none"`` + ``https_only=True``; change it in
  ``create_app`` only when that deployment shape is confirmed, not
  silently for local development.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.middleware.auth_guard import AuthException
from app.routes import api_router, json_api_router


def _frontend_origins() -> list[str]:
    raw = os.getenv("FRONTEND_URL", "http://localhost:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


def create_app() -> FastAPI:
    """Application factory for the SkinCheck API backend."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="SkinCheck API backend (JSON only) for the Next.js frontend",
        version="2.0.0-api-only",
        debug=settings.APP_DEBUG,
    )

    # Session first (added first), CORS second so CORS runs outermost.
    # Starlette runs the most recently added middleware first.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.SECRET_KEY,
        session_cookie="skincheck_session",
        max_age=86400 * 7,  # 7 days, same as the monolith
        same_site="lax",
        https_only=not settings.APP_DEBUG,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_frontend_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response

    @app.exception_handler(AuthException)
    async def auth_exception_handler(request: Request, exc: AuthException):
        # API-only: JSON 401, never an HTML redirect to /login.
        return JSONResponse(status_code=401, content={"detail": "Login required"})

    # JSON routers only. No static mount, no HTML routers.
    app.include_router(json_api_router)
    app.include_router(api_router)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    return app


app = create_app()

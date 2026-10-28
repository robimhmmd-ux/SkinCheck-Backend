"""Backend (API-only) routers.

Only JSON routers are mounted by ``app.main``:
- ``json_api_router`` — /api/v2, the API for the Next.js frontend.
- ``api_router`` — /api/v1/health, kept from the monolith (no templates).

Legacy HTML routers (home_routes, auth_routes, customer_routes,
admin_routes) are still present on disk under this folder as reference,
but they are NOT imported here and NOT mounted, because they render
Jinja templates in ``app/views/`` — and templates are now the frontend
repo's job. Importing them would also pull in the old controllers'
template rendering. Do not re-add them to this file without restoring
the template layer.
"""

from .api_routes import router as api_router
from .json_api import router as json_api_router

__all__ = [
    "json_api_router",
    "api_router",
]

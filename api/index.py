"""Vercel entrypoint for the SkinCheck split backend (FastAPI, API-only).

Vercel treats files under api/ as serverless functions. This file only
re-exports the real app from app.main, so local development (run.py)
and Vercel use the same application object.

Notes for Vercel:
- The database is TiDB Cloud (MySQL-compatible), configured only via
  environment variables in the Vercel dashboard — never via a committed
  .env file. The tables are created with skincheck_setup.sql, not by
  running setup_db.py on Vercel.
- Uploads go to /tmp on Vercel (see app/services/image_service.py),
  because the serverless filesystem is read-only except /tmp.
- Full AI scan analysis is not available here: tensorflow / opencv /
  google-generativeai are lazy imports and are not in the Vercel
  bundle, and the .h5 model is not in the repo. Auth, dashboard,
  history, products and scan upload (status "processing") still work.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402,F401  (re-exported for Vercel)

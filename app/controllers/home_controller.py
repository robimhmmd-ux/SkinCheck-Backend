from fastapi import Request
from fastapi.templating import Jinja2Templates
from app.config import settings

templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))

class HomeController:
    """Controller responsible for public marketing and landing pages."""

    @staticmethod
    async def index(request: Request):
        """Render the main landing page with product features and overview."""
        context = {
            "request": request,
            "page_title": "SkinCheck — AI Skin Health Monitoring & Sunscreen Evenness",
            "app_name": settings.APP_NAME,
        }
        return templates.TemplateResponse("pages/home.html", context)

    @staticmethod
    async def activity_diagram(request: Request):
        """Serve interactive 4-Actor Activity Diagram with download feature."""
        from fastapi.responses import FileResponse
        from pathlib import Path
        diagram_path = Path("activity_diagram_skincheck.html")
        if diagram_path.exists():
            return FileResponse(diagram_path, media_type="text/html")
        return templates.TemplateResponse("pages/home.html", {"request": request, "page_title": "SkinCheck"})

    @staticmethod
    async def usecase_diagram(request: Request):
        """Serve interactive 4-Actor Use Case Diagram (18 use cases)."""
        from fastapi.responses import FileResponse
        from pathlib import Path
        diagram_path = Path("usecase_diagram_skincheck.html")
        if diagram_path.exists():
            return FileResponse(diagram_path, media_type="text/html")
        return templates.TemplateResponse("pages/home.html", {"request": request, "page_title": "SkinCheck"})

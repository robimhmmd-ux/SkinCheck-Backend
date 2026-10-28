import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"==================================================")
    print(f" Starting {settings.APP_NAME} Web Platform (MVC)")
    print(f" Environment : {settings.APP_ENV}")
    print(f" Server URL  : http://{settings.APP_HOST}:{settings.APP_PORT}")
    print(f" Admin Portal: http://{settings.APP_HOST}:{settings.APP_PORT}/admin/dashboard")
    print(f"==================================================")
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG
    )

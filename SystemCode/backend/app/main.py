from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dashboard import router as dashboard_router
from app.core.config import settings
from app.core.exceptions import ResourceNotFoundError


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    dashboard_router,
    prefix="/api/v1",
)


@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_handler(
    request: Request,
    exc: ResourceNotFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )


@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
    }


@app.get(
    "/health",
    tags=["System"],
)
def health():
    return {
        "status": "ok",
    }
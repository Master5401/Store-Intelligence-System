"""
src/api/main.py
────────────────
FastAPI application factory.

  • CORS configured for local dashboard dev
  • All routers mounted
  • DB initialised on startup
  • Static files served for the live dashboard HTML
  • OpenAPI docs available at /docs and /redoc
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.storage.database import init_db

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("SIS API starting — store=%s", settings.store_id)
    await init_db(settings.database_url)
    logger.info("Database ready")
    yield
    logger.info("SIS API shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Store Intelligence System API",
        description=(
            "End-to-end CCTV AI pipeline: motion detection, VSUMM keyframe "
            "extraction, ByteTrack person tracking, pose-based shoplifting "
            "detection, POS integration, and real-time WebSocket streaming."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow dashboard dev server and production origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount routers ─────────────────────────────────────────────────────────
    from src.api.routes.analytics import router as analytics_router
    from src.api.routes.alerts    import router as alerts_router
    from src.api.routes.cameras   import router as cameras_router
    from src.api.routes.pos       import router as pos_router
    from src.api.routes.ws        import auth_router, sys_router, ws_router

    app.include_router(auth_router)
    app.include_router(sys_router)
    app.include_router(analytics_router)
    app.include_router(alerts_router)
    app.include_router(cameras_router)
    app.include_router(pos_router)
    app.include_router(ws_router)          # WebSocket (no prefix)

    # ── Static files (dashboard HTML) ─────────────────────────────────────────
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_redirect():
        return """
        <html><head><meta http-equiv="refresh" content="0; url=/static/dashboard.html"></head>
        <body><a href="/static/dashboard.html">Open Dashboard</a></body></html>
        """

    return app


app = create_app()

from __future__ import annotations

from fastapi import FastAPI

from .routes import build_router
from .runtime import ServerConfig


def create_app(server_config: ServerConfig) -> FastAPI:
    app = FastAPI(title="rank-llm", version="0.0.1")
    app.include_router(build_router(server_config))
    return app

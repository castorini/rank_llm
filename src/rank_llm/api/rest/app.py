from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .routes import build_router
from .runtime import ServerConfig, validation_error_response


def create_app(server_config: ServerConfig) -> FastAPI:
    app = FastAPI(title="rank-llm", version="0.0.1")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            validation_error_response(str(error)).to_envelope(), status_code=400
        )

    app.include_router(build_router(server_config))
    return app

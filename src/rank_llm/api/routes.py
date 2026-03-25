from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .runtime import (
    ServerConfig,
    run_rerank_request,
    runtime_error_response,
    validation_error_response,
)


def build_router(config: ServerConfig) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/v1/rerank")
    def rerank(payload: dict[str, Any]) -> JSONResponse:
        try:
            response = run_rerank_request(payload, config=config)
            return JSONResponse(response.to_envelope())
        except (TypeError, ValueError, KeyError) as error:
            response = validation_error_response(str(error))
            return JSONResponse(response.to_envelope(), status_code=400)
        except Exception as error:  # noqa: BLE001
            response = runtime_error_response(error)
            return JSONResponse(response.to_envelope(), status_code=500)

    return router

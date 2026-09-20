from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from rank_llm.api.options import RerankValidationError, request_schema

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

    def execute(payload: dict[str, Any], *, retrieval: bool = False) -> JSONResponse:
        try:
            response = run_rerank_request(payload, config=config, retrieval=retrieval)
            return JSONResponse(response.to_envelope())
        except RerankValidationError as error:
            return JSONResponse(
                validation_error_response(str(error)).to_envelope(), status_code=400
            )
        except Exception as error:  # noqa: BLE001
            response = runtime_error_response(error)
            return JSONResponse(
                response.to_envelope(),
                status_code=502 if response.status == "provider_error" else 500,
            )

    @router.post("/v1/rerank")
    def rerank(
        payload: Annotated[dict[str, Any], Body(json_schema_extra=request_schema())],
    ) -> JSONResponse:
        return execute(payload)

    @router.post("/v1/retrieve-and-rerank")
    def retrieve_and_rerank(
        payload: Annotated[
            dict[str, Any], Body(json_schema_extra=request_schema(retrieval=True))
        ],
    ) -> JSONResponse:
        return execute(payload, retrieval=True)

    return router

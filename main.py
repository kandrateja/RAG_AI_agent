"""FastAPI application entry point."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.routes import router
from app.config import get_settings


def _downgrade_binary_fields(schema: dict[str, Any]) -> None:
    """Walk the OpenAPI schema and rewrite OpenAPI 3.1's ``contentMediaType``
    binary hints back to OpenAPI 3.0's ``format: binary``.

    Swagger UI's file-picker widget keys off ``format: binary``. Without
    this rewrite, an ``UploadFile`` field renders as a plain string
    textbox ("Add string item"), not an upload button.
    """
    if isinstance(schema, dict):
        if (
            schema.get("type") == "string"
            and schema.get("contentMediaType") == "application/octet-stream"
        ):
            schema.pop("contentMediaType", None)
            schema["format"] = "binary"
        for v in schema.values():
            _downgrade_binary_fields(v)
    elif isinstance(schema, list):
        for v in schema:
            _downgrade_binary_fields(v)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(
        title=settings.api_title,
        version="0.1.0",
        description=(
            "Ingestion + Q&A + drafting API for DIMS regulatory letter packages. "
            "Sources: file-store folders or direct upload. Extraction: IBM Docling. "
            "LLM: Anthropic Claude Opus on Google Vertex AI."
        ),
    )
    app.include_router(router)

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["openapi"] = "3.0.2"
        _downgrade_binary_fields(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


app = create_app()

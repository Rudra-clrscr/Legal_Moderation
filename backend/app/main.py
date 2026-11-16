"""FastAPI entrypoint: ``uvicorn app.main:app`` (run from the ``backend`` directory)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.errors import LegalBotError

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


async def _warm_up(container: Container) -> None:
    """Load the LLM and embedding model in the background so the first request is fast."""
    try:
        await asyncio.to_thread(container.vector_store.embedder.warm_up)
    except Exception:
        logger.warning("Embedding model warm-up failed", exc_info=True)
    try:
        await asyncio.to_thread(container.llm.load)
    except LegalBotError as exc:
        logger.warning("LLM not loaded: %s", exc.message)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(settings or get_settings())
        app.state.container = c
        c.registry.recover_interrupted()
        warm_task = None
        if container is None and c.settings.llm_load_on_startup:
            warm_task = asyncio.create_task(_warm_up(c))
        yield
        if warm_task and not warm_task.done():
            warm_task.cancel()
        if container is None:
            c.llm.close()  # stop the LLM worker process

    app = FastAPI(title="Legal Document Analysis API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(LegalBotError)
    async def handle_domain_error(_: Request, exc: LegalBotError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

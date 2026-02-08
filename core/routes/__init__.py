"""
core/routes/__init__.py

Router aggregation — registers all route modules on the FastAPI app.
"""

from fastapi import FastAPI

from .chat import router as chat_router
from .state import router as state_router
from .memory import router as memory_router
from .notes import router as notes_router
from .worker import router as worker_router
from .settings import router as settings_router


def setup_routes(app: FastAPI) -> None:
    """Include all route modules."""
    app.include_router(chat_router)
    app.include_router(state_router)
    app.include_router(memory_router)
    app.include_router(notes_router)
    app.include_router(worker_router)
    app.include_router(settings_router)

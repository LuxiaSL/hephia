"""
core/routes/worker.py

Worker task queue endpoints — submit and poll async tasks.
"""

from fastapi import APIRouter, Request, HTTPException, Body

from shared_models.api_models import (
    WorkerTaskRequest,
    WorkerTaskResponse,
    WorkerTaskStatus,
)
from loggers import SystemLogger

router = APIRouter(prefix="/worker", tags=["worker"])


@router.post("/task", response_model=WorkerTaskResponse, status_code=202)
async def submit_task(request: Request, body: WorkerTaskRequest = Body(...)):
    """Submit a task for async worker execution."""
    queue = request.app.state.worker_queue
    try:
        task_id = queue.submit(task_spec=body.task, context=body.context)
        return WorkerTaskResponse(task_id=task_id)
    except Exception as e:
        SystemLogger.error(f"Worker submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=WorkerTaskStatus)
async def get_task_status(task_id: str, request: Request):
    """Get the status of a submitted worker task."""
    queue = request.app.state.worker_queue
    try:
        status = queue.get_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Worker status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
core/routes/worker.py

Worker task queue endpoints — submit, poll, and reply to async tasks.
"""

from fastapi import APIRouter, Request, HTTPException, Body

from shared_models.api_models import (
    WorkerClearRequest,
    WorkerReplyRequest,
    WorkerStarResponse,
    WorkerTaskRequest,
    WorkerTaskResponse,
    WorkerTaskStatus,
)
from loggers import SystemLogger

router = APIRouter(prefix="/worker", tags=["worker"])


@router.post("/task", response_model=WorkerTaskResponse, status_code=202)
async def submit_task(request: Request, body: WorkerTaskRequest = Body(...)):
    """Submit a task for async worker execution."""
    SystemLogger.info(f"Worker route: received task submission — {body.task[:80]}")
    queue = request.app.state.worker_queue
    try:
        task_id = queue.submit(task_spec=body.task, context=body.context)
        SystemLogger.info(f"Worker route: task submitted — {task_id}")
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


@router.post("/{task_id}/reply", response_model=WorkerTaskStatus)
async def reply_to_task(
    task_id: str, request: Request, body: WorkerReplyRequest = Body(...)
):
    """Send a reply to an active worker task (answer a question or follow up)."""
    queue = request.app.state.worker_queue
    try:
        status = await queue.reply(task_id=task_id, message=body.message)
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        SystemLogger.error(f"Worker reply error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{task_id}/stop", response_model=WorkerTaskStatus)
async def stop_task(task_id: str, request: Request):
    """Stop/interrupt a running worker task."""
    queue = request.app.state.worker_queue
    try:
        status = await queue.stop(task_id=task_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        SystemLogger.error(f"Worker stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{task_id}")
async def delete_task(task_id: str, request: Request):
    """Delete a completed/failed worker task."""
    queue = request.app.state.worker_queue
    try:
        found = await queue.delete_task(task_id)
        if not found:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Worker delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_tasks(request: Request, body: WorkerClearRequest = Body(...)):
    """Bulk remove completed (and optionally failed) tasks."""
    queue = request.app.state.worker_queue
    try:
        removed = await queue.clear_tasks(completed_only=body.completed_only)
        return {"removed": removed}
    except Exception as e:
        SystemLogger.error(f"Worker clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{task_id}/star", response_model=WorkerStarResponse)
async def star_task(task_id: str, request: Request):
    """Star a completed task — saves its result as a cognitive memory."""
    queue = request.app.state.worker_queue
    mind = request.app.state.mind
    try:
        status = queue.get_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if status.status != "completed" or not status.result:
            raise HTTPException(
                status_code=400,
                detail="Can only star completed tasks with results",
            )

        # Truncate to 1000 chars for memory formation
        content = f"Worker task result: {status.result[:1000]}"
        node_id = await mind.memory_formation.process_explicit_memory(
            content=content,
            source="worker_task",
        )

        if node_id:
            return WorkerStarResponse(node_id=node_id)
        else:
            return WorkerStarResponse(
                error="Memory formed but below significance threshold",
            )
    except HTTPException:
        raise
    except Exception as e:
        SystemLogger.error(f"Worker star error: {e}")
        return WorkerStarResponse(error=str(e))

"""
core/routes/state.py

State and action endpoints.
"""

from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException

from shared_models.api_models import (
    ActionRequest,
    ActionResponse,
    ActionStatus,
    ActionInfo,
)
from event_dispatcher import global_event_dispatcher, Event

router = APIRouter()


def _flatten_action_status(raw_status: dict) -> dict:
    """Flatten nested action status dict for ActionStatus model."""
    history = raw_status.get("history", {})
    return {
        "last_execution": history.get("last_execution", 0),
        "total_executions": history.get("total_executions", 0),
        "successful_executions": history.get("successful_executions", 0),
        "failed_executions": history.get("failed_executions", 0),
        "on_cooldown": raw_status.get("on_cooldown", False),
        "remaining_cooldown": raw_status.get("remaining_cooldown", 0),
    }


@router.get("/v1/actions/state")
async def get_state(request: Request):
    """Get current system state."""
    state_bridge = request.app.state.state_bridge
    return await state_bridge.get_api_context(use_memory_emotions=False)


@router.get("/state")
async def get_state_alias(request: Request):
    """Get current system state (simple alias)."""
    state_bridge = request.app.state.state_bridge
    return await state_bridge.get_api_context(use_memory_emotions=False)


@router.post("/v1/actions/{action_name}")
async def execute_action(action_name: str, request: Request, body: ActionRequest) -> ActionResponse:
    """Execute a specific action."""
    internal = request.app.state.internal
    try:
        if action_name not in internal.action_manager.available_actions:
            raise HTTPException(status_code=404, detail=f"Action '{action_name}' not found")

        result = internal.action_manager.perform_action(action_name, **body.parameters)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Action failed"))

        global_event_dispatcher.dispatch_event(Event("internal:action", {
            "action": action_name,
            "result": result,
        }))

        return ActionResponse(
            success=True,
            message=f"Successfully executed {action_name}",
            state_changes=result.get("state_changes", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/actions")
async def get_available_actions(request: Request) -> Dict[str, ActionInfo]:
    """Get information about all available actions and their current status."""
    internal = request.app.state.internal
    try:
        actions = internal.action_manager.available_actions
        statuses = internal.action_manager.get_action_status()

        response: Dict[str, ActionInfo] = {}
        for name, action in actions.items():
            parameters: Dict[str, Any] = {}
            if hasattr(action, "_base_cooldown"):
                parameters["cooldown"] = action._base_cooldown
            if hasattr(action, "calculate_recovery_amount"):
                parameters["dynamic_recovery"] = True

            raw_status = statuses.get(name, {})
            flat_status = _flatten_action_status(raw_status)
            status_obj = ActionStatus(**flat_status)

            response[name] = ActionInfo(
                name=name,
                description=action.__doc__ or "No description available",
                parameters=parameters,
                status=status_obj,
            )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/actions/{action_name}/status")
async def get_action_status(action_name: str, request: Request) -> ActionStatus:
    """Get status of a specific action."""
    internal = request.app.state.internal
    try:
        if action_name not in internal.action_manager.available_actions:
            raise HTTPException(status_code=404, detail=f"Action '{action_name}' not found")
        raw_status = internal.action_manager.get_action_status(action_name)
        flat_status = _flatten_action_status(raw_status)
        return ActionStatus(**flat_status)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

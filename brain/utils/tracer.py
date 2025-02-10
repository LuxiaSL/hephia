"""
brain/utils/tracer.py

Enhanced tracing system for brain components, providing detailed error tracking
and performance monitoring.
"""

import inspect
import dataclasses
import functools
import asyncio
from typing import Dict, Any, Optional, Callable, Set, List
from datetime import datetime
from enum import Enum

from loggers import BrainLogger

class TraceLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def to_traceable(value: Any) -> Dict[str, Any]:
    """Convert various data types to a traceable format."""
    try:
        if dataclasses.is_dataclass(value):
            return {
                f"{value.__class__.__name__}.{k}": str(v)
                for k, v in dataclasses.asdict(value).items()
            }
        elif isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return {str(i): str(v) for i, v in enumerate(value)}
        elif hasattr(value, '__dict__'):
            return {
                f"{value.__class__.__name__}.{k}": str(v)
                for k, v in value.__dict__.items()
            }
        else:
            return {"value": str(value)}
    except Exception as e:
        return {"error": f"Failed to trace value: {str(e)}"}

class TraceContext:
    """Manages trace context and state."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.error: Optional[Exception] = None
        
    def add_event(self, name: str, attributes: Dict[str, Any]):
        """Add an event to the trace context."""
        self.events.append({
            "name": name,
            "time": datetime.now(),
            "attributes": attributes
        })
        
    def set_error(self, error: Exception):
        """Record an error in the trace context."""
        import traceback
        self.error = error
        self.attributes["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }

class BrainTracer:
    """Main tracing functionality for brain components."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._active_traces: Dict[str, TraceContext] = {}
        self._ignored_attributes: Set[str] = {"password", "token", "secret"}
        
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return self._wrap_function(args[0])
        else:
            return self._handle_event(*args, **kwargs)
            
    def _wrap_function(self, func: Callable) -> Callable:
        """Wrap a function with tracing."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = TraceContext(func.__qualname__)
            self._active_traces[func.__qualname__] = context
            
            try:
                # Capture and trace arguments
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filter sensitive data
                traced_args = {
                    k: to_traceable(v) if k not in self._ignored_attributes else "<redacted>"
                    for k, v in bound_args.arguments.items()
                }
                context.attributes["arguments"] = traced_args
                
                # Execute function
                BrainLogger.debug(f"Entering {func.__qualname__}")
                result = await func(*args, **kwargs)
                
                # Trace result
                if result is not None:
                    context.attributes["result"] = to_traceable(result)
                
                return result
                
            except Exception as e:
                context.set_error(e)
                BrainLogger.error(
                    f"Error in {func.__qualname__}:\n"
                    f"Arguments: {context.attributes.get('arguments', {})}\n"
                    f"Error: {context.attributes['error']}"
                )
                raise
                
            finally:
                duration = datetime.now() - context.start_time
                context.attributes["duration_ms"] = duration.total_seconds() * 1000
                self._active_traces.pop(func.__qualname__, None)
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = TraceContext(func.__qualname__)
            self._active_traces[func.__qualname__] = context
            
            try:
                # Similar to async_wrapper but synchronous
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                traced_args = {
                    k: to_traceable(v) if k not in self._ignored_attributes else "<redacted>"
                    for k, v in bound_args.arguments.items()
                }
                context.attributes["arguments"] = traced_args
                
                BrainLogger.debug(f"Entering {func.__qualname__}")
                result = func(*args, **kwargs)
                
                if result is not None:
                    context.attributes["result"] = to_traceable(result)
                
                return result
                
            except Exception as e:
                context.set_error(e)
                BrainLogger.error(
                    f"Error in {func.__qualname__}:\n"
                    f"Arguments: {context.attributes.get('arguments', {})}\n"
                    f"Error: {context.attributes['error']}"
                )
                raise
                
            finally:
                duration = datetime.now() - context.start_time
                context.attributes["duration_ms"] = duration.total_seconds() * 1000
                self._active_traces.pop(func.__qualname__, None)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def _handle_event(self, *args, **kwargs):
        """Handle trace events and attributes."""
        current_trace = next(iter(self._active_traces.values()), None)
        if not current_trace:
            return
            
        if "attr" in kwargs and kwargs["attr"]:
            current_trace.attributes[self.name] = args[0]
            return args[0]
        else:
            current_trace.add_event(
                self.name,
                {"args": to_traceable(args), "kwargs": to_traceable(kwargs)}
            )
            return args[0] if len(args) == 1 else args if args else None
            
    def __getattr__(self, name: str) -> 'BrainTracer':
        """Allow chained attribute access for event names."""
        if self.name is None:
            return BrainTracer(name)
        return BrainTracer(f"{self.name}.{name}")

# Create global instance
brain_trace = BrainTracer()
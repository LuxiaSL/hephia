# brain/interfaces/exo_utils/hud/sections/system.py

from datetime import datetime
from typing import Dict, Any

from .base import BaseHudSection
from config import Config
from loggers import BrainLogger

# Attempt to import psutil and set a flag
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    BrainLogger.info("HUD: psutil library not found. System resource monitoring (CPU/Mem) will be disabled.")


class SystemHudSection(BaseHudSection):
    """
    HUD Section for system-level information like time, date,
    simulated system "weather" (CPU/memory), and turn pacing.
    """

    def __init__(self, prompt_key: str = 'interfaces.exo.hud.system', section_name: str = "System"):
        super().__init__(prompt_key=prompt_key, section_name=section_name)

    async def _prepare_prompt_vars(self, hud_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepares variables for the system information HUD section.
        """
        system_vars = {
            "hud_header_str": f"[{self.section_name}]",
            # Time and Date
            "system_current_date_str": "N/A",
            "system_current_time_str": "N/A",
            "system_day_of_week_str": "N/A",
            "system_timezone_str": "N/A", 

            # System "Weather" (Resources)
            "system_cpu_load_percent_str": "N/A",
            "system_memory_usage_percent_str": "N/A",
            "system_cpu_temp_celsius_str": "N/A", # Optional, often unavailable

            # Turn Pacing
            "system_turn_pacing_actual_seconds_str": "N/A",
            "system_turn_pacing_expected_seconds_str": "N/A",

            # Conversation Limit (Handled by ExoProcessorInterface populating hud_metadata)
            "system_conversation_fill_percent_str": "N/A", # Populated from hud_metadata
            "system_turn_pacing_status_str": "N/A", 
        }

        # 1. Time and Date
        # hud_metadata should contain 'current_interaction_time'
        current_time = hud_metadata.get("current_interaction_time", datetime.now())
        system_vars["system_current_date_str"] = current_time.strftime("%Y-%m-%d")
        system_vars["system_current_time_str"] = current_time.strftime("%H:%M:%S")
        system_vars["system_day_of_week_str"] = current_time.strftime("%A")
        try:
            # Get local timezone name if possible (can be unreliable)
            system_vars["system_timezone_str"] = current_time.astimezone().tzname() or "Local"
        except Exception:
            system_vars["system_timezone_str"] = "Local"


        # 2. System "Weather" (CPU/Memory) using psutil if available
        if PSUTIL_AVAILABLE:
            try:
                cpu_load = psutil.cpu_percent(interval=None) # Non-blocking, gets overall CPU load
                system_vars["system_cpu_load_percent_str"] = f"{cpu_load:.1f}%"
            except Exception as e:
                BrainLogger.debug(f"HUD ({self.section_name}): Could not get CPU load: {e}")
                system_vars["system_cpu_load_percent_str"] = "Err"

            try:
                memory_info = psutil.virtual_memory()
                system_vars["system_memory_usage_percent_str"] = f"{memory_info.percent:.1f}%"
            except Exception as e:
                BrainLogger.debug(f"HUD ({self.section_name}): Could not get memory usage: {e}")
                system_vars["system_memory_usage_percent_str"] = "Err"
            
            # CPU temperature is more complex and OS-dependent
            try:
                temps = psutil.sensors_temperatures()
                # Find a core temperature or general package temperature
                # This logic might need adjustment based on typical output of psutil.sensors_temperatures()
                cpu_temp_to_report = None
                if temps: # temps is a dict like {'coretemp': [shwtemp(label='Package id 0', current=63.0, high=100.0, critical=100.0), ...]}
                    for key in ['coretemp', 'k10temp', 'acpitz', 'cpu_thermal']:
                        if key in temps:
                            for entry in temps[key]:
                                if 'current' in entry._fields and ('core' in entry.label.lower() or 'package' in entry.label.lower() or not entry.label): # Prioritize core/package or unlabeled
                                    cpu_temp_to_report = entry.current
                                    break
                            if cpu_temp_to_report is not None:
                                break
                if cpu_temp_to_report is not None:
                    system_vars["system_cpu_temp_celsius_str"] = f"{cpu_temp_to_report:.1f}°C"
                else:
                    system_vars["system_cpu_temp_celsius_str"] = "N/A" # Not found or not supported
            except AttributeError: # sensors_temperatures might not exist on all OSes
                system_vars["system_cpu_temp_celsius_str"] = "N/S" # Not Supported
            except Exception as e:
                BrainLogger.debug(f"HUD ({self.section_name}): Could not get CPU temperature: {e}")
                system_vars["system_cpu_temp_celsius_str"] = "Err"
        else: # psutil not available
            system_vars["system_cpu_load_percent_str"] = "N/A (psutil missing)"
            system_vars["system_memory_usage_percent_str"] = "N/A (psutil missing)"
            system_vars["system_cpu_temp_celsius_str"] = "N/A (psutil missing)"


        # 3. Turn Pacing
        last_turn_time = hud_metadata.get("last_interaction_time")
        expected_interval_seconds = Config.get_exo_min_interval()
        system_vars["system_turn_pacing_expected_seconds_str"] = f"{expected_interval_seconds:.2f}s"

        if last_turn_time and isinstance(last_turn_time, datetime):
            # current_time is already defined above for date/time section
            actual_interval_delta = current_time - last_turn_time
            actual_interval_seconds = actual_interval_delta.total_seconds()
            system_vars["system_turn_pacing_actual_seconds_str"] = f"{actual_interval_seconds:.2f}s"

            # Compare actual to expected (with a small tolerance)
            tolerance = expected_interval_seconds * 0.1 # 10% tolerance
            if actual_interval_seconds < expected_interval_seconds - tolerance:
                system_vars["system_turn_pacing_status_str"] = "Fast"
            elif actual_interval_seconds > expected_interval_seconds + tolerance:
                system_vars["system_turn_pacing_status_str"] = "Slow"
            else:
                system_vars["system_turn_pacing_status_str"] = "Nominal"
        else:
            system_vars["system_turn_pacing_actual_seconds_str"] = "N/A"
            system_vars["system_turn_pacing_status_str"] = "First turn"
            
        # 4. Conversation Limit Info (Data provided by ExoProcessorInterface in hud_metadata)
        conv_fill_percent = (hud_metadata.get("conversation_state_size") / Config.get_exo_max_turns()) * 100 if hud_metadata.get("conversation_state_size") is not None else None
        if conv_fill_percent is not None:
            if conv_fill_percent >= 100:
                system_vars["system_conversation_fill_percent_str"] = f"{conv_fill_percent:.0f}% full, trimming active..."
            else:
                system_vars["system_conversation_fill_percent_str"] = f"{conv_fill_percent:.0f}% full"
        else:
            system_vars["system_conversation_fill_percent_str"] = "N/A"


        return system_vars
# client/config/utils.py
"""
File I/O utilities for the Hephia Configuration TUI.
Handles reading and writing .env, models.json, and YAML prompt files.
"""
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml
from dotenv import dotenv_values, set_key, unset_key

PROJECT_ROOT = Path(".") # This should ideally be more robustly determined.
DOTENV_PATH = PROJECT_ROOT / ".env"

def _create_backup(file_path: Path) -> bool:
    """Creates a .bak copy of the given file. Returns True if successful/backup made."""
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        try:
            shutil.copy2(file_path, backup_path)
            # print(f"Backup created: {backup_path}") # For server-side logging if needed
            return True
        except Exception as e:
            print(f"Warning: Could not create backup for {file_path}: {e}") # Log this
    return False

def backup_dotenv_file() -> bool:
    """Explicitly backs up the .env file."""
    # Ensure .env exists before trying to back it up, otherwise shutil.copy2 fails
    if not DOTENV_PATH.exists():
        DOTENV_PATH.touch() # Create if it doesn't exist to prevent error on first save
    return _create_backup(DOTENV_PATH)

def load_dotenv_values() -> Dict[str, Optional[str]]:
    """Loads key-value pairs from the .env file."""
    # Create .env if it doesn't exist, otherwise dotenv_values might have issues
    if not DOTENV_PATH.exists():
        DOTENV_PATH.touch()
    return dotenv_values(DOTENV_PATH)

def _should_quote_value(value: str) -> bool:
    """
    Determine if a value should be quoted based on content.
    Only quote if the value contains spaces, special characters, or starts with quotes.
    """
    if not value:
        return False
    
    # Don't quote simple alphanumeric values, booleans, or numbers
    if value.isalnum():
        return False
    
    # Don't quote simple boolean strings or numeric strings
    if value.lower() in ('true', 'false') or value.replace('.', '').replace('-', '').isdigit():
        return False
    
    # Quote if contains spaces or special shell characters
    special_chars = {' ', '\t', '\n', '"', "'", '\\', '$', '`', '!', '*', '?', '[', ']', '(', ')', '{', '}', ';', '&', '|', '<', '>'}
    return any(char in special_chars for char in value)

def save_dotenv_value(key: str, value: Optional[str]):
    """
    Saves a single key-value pair to the .env file.
    If value is None, the key is removed.
    Uses smart quoting to avoid unnecessary quotes around simple values.
    """
    # Ensure .env exists before trying to modify it.
    if not DOTENV_PATH.exists():
        DOTENV_PATH.touch()
        
    try:
        if value is None:
            unset_key(DOTENV_PATH, key)
        else:
            # Determine quote mode based on content
            quote_mode = "auto" if _should_quote_value(value) else "never"
            set_key(DOTENV_PATH, key, value, quote_mode=quote_mode)
    except Exception as e: # Catch potential IOErrors or other issues from python-dotenv
        print(f"Error saving .env key '{key}': {e}")
        raise # Re-raise for the TUI to handle

# --- models.json File Handling ---
def get_models_json_path() -> Path:
    """
    Determines the path to the models.json file based on OS.
    Mirrors logic from Config.load_user_models in the main project.
    """
    if platform.system() == "Windows":
        path = Path(os.getenv("APPDATA", str(Path.home()))) / "hephia" / "models.json"
    else:
        path = Path.home() / ".config" / "hephia" / "models.json"
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

MODELS_JSON_PATH = get_models_json_path()

def ensure_models_json_exists() -> None:
    """Ensures the models.json file exists, creating an empty one if necessary."""
    if not MODELS_JSON_PATH.exists():
        MODELS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty JSON file
        try:
            with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not create models.json: {e}")

def get_all_available_model_names() -> List[Tuple[str, str]]:
    """
    Gets all model names for selection, merging AVAILABLE_MODELS from config.py
    with custom models from models.json.
    Returns a list of (display_name, model_id_or_key) tuples.
    """
    choices_dict: Dict[str, str] = {}
    
    try:
        from config import Config as MainConfig #
        if hasattr(MainConfig, 'AVAILABLE_MODELS'): #
            for name in MainConfig.AVAILABLE_MODELS.keys(): #
                if name not in choices_dict:
                    choices_dict[name] = name
    except ImportError:
        # This TUI is part of the main project, so this should ideally not fail.
        # If it does, it indicates a problem with how the TUI is launched or sys.path.
        print("CRITICAL WARNING: Main project 'config.py' could not be imported in get_all_available_model_names.")
        # Potentially raise an error or return empty if this is critical for TUI operation
                
    try:
        # load_models_json might raise RuntimeError if file is corrupt/unreadable
        user_models_data = load_models_json() #
        if isinstance(user_models_data, dict):
            for name in user_models_data.keys():
                if name not in choices_dict:
                    choices_dict[name] = name
    except RuntimeError as e: 
        # If models.json is missing or corrupt, we can still proceed with default models.
        # The TUI screen responsible for models.json will handle showing errors for that file.
        print(f"Info: Could not load user models for model list: {e}")
    except Exception as e_gen: # Catch any other unexpected error during models.json loading
        print(f"Warning: Unexpected error loading user models for model list: {e_gen}")
                
    choices_list = [(name, model_key) for name, model_key in choices_dict.items()]

    return sorted(choices_list)

def load_models_json() -> Dict[str, Any]:
    """Loads the user's custom models from models.json."""
    ensure_models_json_exists()  # Ensure file exists
    
    try:
        with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}  # Return empty dict for empty file
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: models.json contains invalid JSON: {e}")
        raise RuntimeError(f"models.json contains invalid JSON: {e}")
    except IOError as e:
        print(f"Error: Could not read models.json: {e}")
        raise RuntimeError(f"Failed to read models.json: {e}")

def save_models_json(models_data: Dict[str, Any]) -> bool:
    """Saves the models data to models.json, creating a backup first. Returns True if backup was made."""
    MODELS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODELS_JSON_PATH.exists():
         MODELS_JSON_PATH.touch() # Create if doesn't exist for backup

    backup_made = _create_backup(MODELS_JSON_PATH)
    try:
        with open(MODELS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(models_data, f, indent=4)
        return backup_made
    except IOError as e:
        print(f"Error saving models.json: {e}")
        raise # Re-raise for TUI to handle

# --- YAML Prompt File Handling ---
def load_yaml_file(file_path: Path) -> Optional[Any]:
    """Loads a YAML file from the given path."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, IOError) as e:
        print(f"Error loading YAML file {file_path}: {e}")
        raise RuntimeError(f"Failed to load YAML {file_path.name}: {e}")

def save_yaml_file(file_path: Path, data: Any) -> bool:
    """
    Saves data to a YAML file with proper multiline formatting.
    Preserves block scalar style for multiline strings.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.touch()

    backup_made = _create_backup(file_path)
    
    try:
        # Custom YAML dumper that preserves multiline formatting
        class CustomYAMLDumper(yaml.SafeDumper):
            def write_literal(self, text):
                # Force literal style (|) for multiline strings
                return super().write_literal(text)
            
            def represent_str(self, data):
                # Use literal block style for multiline strings
                if '\n' in data or len(data) > 50:
                    return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                return self.represent_scalar('tag:yaml.org,2002:str', data)
        
        # Add the custom string representer
        CustomYAMLDumper.add_representer(str, CustomYAMLDumper.represent_str)
        
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, 
                Dumper=CustomYAMLDumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
                width=float('inf')  # Don't wrap long lines
            )
        return backup_made
    except (yaml.YAMLError, IOError) as e:
        print(f"Error saving YAML file {file_path}: {e}")
        raise
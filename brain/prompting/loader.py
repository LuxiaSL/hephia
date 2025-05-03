'''
loader.py

Utility for loading and rendering prompt templates from YAML files,
with support for defaults, per-model overrides, and micro-fragment codes.

Usage:
    from utils.prompt_loader import get_prompt, get_code

    # Load a system prompt for Exo summary
    sys = get_prompt(
        key="interfaces.exo.summary.system",
        model=Config.get_summary_model()
    )

    # Load a single-template prompt (e.g. exo memory)
    mem = get_prompt(
        key="interfaces.exo.memory.template",
        model=Config.get_cognitive_model(),
        vars={
            "command_input": cmd_in,
            "content": resp,
            "result_message": res_msg
        }
    )

    # Load a micro-fragment (error/success code)
    err = get_code(
        "COMMAND_VALIDATION_MISSING_FLAG",
        vars={"flag": "limit", "command": "search"},
        default="Flag '--${flag}' is required for command '${command}'."
    )
    raise ValueError(err)
'''  
import os, yaml, string, pathlib, sys, platform
from functools import lru_cache

EXTRA_PATHS = []
if env := os.getenv("HEPHIA_PROMPT_PATHS"):
    EXTRA_PATHS.extend(env.split(os.pathsep))

# XDG / APPDATA fallback
home_cfg = (
    pathlib.Path.home() / ".config" / "hephia" / "prompts"
    if platform.system() != "Windows"
    else pathlib.Path(os.getenv("APPDATA", pathlib.Path.home())) / "hephia" / "prompts"
)
EXTRA_PATHS.append(str(home_cfg))

PROMPT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "prompts")
)

SEARCH_PATHS = [p for p in EXTRA_PATHS if os.path.isdir(p)] + [PROMPT_ROOT]

@lru_cache(maxsize=None)
def _load_yaml(rel_path: str) -> dict:
    """Load and parse a YAML file by searching multiple prompt directories."""
    for root in SEARCH_PATHS:
        full_path = os.path.join(root, rel_path)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error parsing YAML {full_path}: {e}")

    raise FileNotFoundError(f"Prompt file not found in any path: {rel_path}")


def get_prompt(key: str, *, model: str, vars: dict | None = None) -> str:
    """
    Retrieve and render a prompt by key, merging defaults with any per-model overrides.

    Args:
        key: Dot-delimited path to a prompt leaf, e.g.
             'interfaces.exo.summary.system' or 'interfaces.exo.memory.template'.
        model: Model name (corresponding to config values) for looking up overrides.
        vars: Mapping of placeholder names to values for substitution.

    Returns:
        Rendered prompt text.

    Raises:
        FileNotFoundError: If the YAML file is missing.
        KeyError: If the requested section/template is not defined.
        RuntimeError: On YAML parsing errors.
    """
    # Split key into path + leaf
    parts = key.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid prompt key: '{key}'")
    yaml_path = os.path.join(*parts[:-1]) + ".yaml"
    leaf = parts[-1]

    data = _load_yaml(yaml_path)
    defaults = data.get('defaults', {})
    models  = data.get('models', {})

    # Start from defaults and overlay model-specific block if present
    merged = defaults.copy()
    if model in models:
        override = models[model]
        merged.update(override)

    # Extract text
    if 'sections' in merged:
        sections = merged['sections']
        if leaf not in sections:
            raise KeyError(f"Section '{leaf}' not found in {yaml_path}")
        text = sections[leaf]
    elif 'template' in merged and leaf == 'template':
        text = merged['template']
    elif leaf in merged:
        text = merged[leaf]
    else:
        raise KeyError(f"Template key '{leaf}' not found in {yaml_path}")

    # Substitute variables
    try:
        return string.Template(text).safe_substitute(vars or {})
    except Exception as e:
        raise RuntimeError(f"Error substituting variables in prompt '{key}': {e}")


def get_code(code_id: str, *, vars: dict | None = None, default: str | None = None, strict: bool = False) -> str:
    """
    Retrieve a micro-fragment (error or success code) by its ID.

    Args:
        code_id: Identifier from codes/errors.yaml or codes/successes.yaml.
        vars: Placeholder values for substitution.
        default: Fallback template if the code_id isn't found.
        strict: If True, raise KeyError when code_id is missing.
    Returns:
        Rendered micro-fragment.
    Raises:
        KeyError: If strict=True and code_id is not defined.
    """
    # Load both categories
    err_defs = _load_yaml('codes/errors.yaml').get('defaults', {})
    ok_defs  = _load_yaml('codes/successes.yaml').get('defaults', {})
    codes = {**err_defs, **ok_defs}

    if code_id in codes:
        text = codes[code_id]
    else:
        if strict:
            raise KeyError(f"Code ID not found: {code_id}")
        text = default or code_id

    try:
        return string.Template(text).safe_substitute(vars or {})
    except Exception as e:
        raise RuntimeError(f"Error in code '{code_id}' substitution: {e}")

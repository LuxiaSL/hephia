# client/config/screens/prompt_editor.py
"""
Textual screen for editing YAML prompt templates.
Focuses on editing user override files, using base prompts as read-only references.
"""
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Set

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button, Tree, Label, Select, Static, TextArea
)
from textual.widgets.tree import TreeNode
from textual.message import Message

from brain.prompting.loader import PROMPT_ROOT
from ..utils import load_yaml_file, save_yaml_file, get_all_available_model_names
from ..dialogs import ConfirmationDialog

class PromptStructure:
    """Handles YAML structure validation and navigation."""
    
    def __init__(self, yaml_data: Dict[str, Any]):
        self.data = yaml_data or {}
        self.available_leaves = self._discover_leaves()
        self.parameters = self._extract_all_parameters()
    
    def _discover_leaves(self) -> List[str]:
        """Reliably discovers available leaves from defaults section."""
        leaves = []
        defaults = self.data.get('defaults', {})
        
        if not isinstance(defaults, dict):
            return leaves
        
        # Handle sections structure first
        if 'sections' in defaults and isinstance(defaults['sections'], dict):
            for key, value in defaults['sections'].items():
                if isinstance(value, str):
                    leaves.append(key)
        
        # Handle direct string values (but avoid duplicating sections content)
        for key, value in defaults.items():
            if key != 'sections' and isinstance(value, str):
                # Only add if not already found in sections
                if key not in leaves:
                    leaves.append(key)
        
        return sorted(list(set(leaves)))
    
    def _extract_all_parameters(self) -> Set[str]:
        """Extract all ${param} parameters from all templates."""
        params = set()
        
        def scan_value(value):
            if isinstance(value, str):
                found = re.findall(r'\$\{(\w+)\}', value)
                params.update(found)
            elif isinstance(value, dict):
                for v in value.values():
                    scan_value(v)
            elif isinstance(value, list):
                for item in value:
                    scan_value(item)
        
        scan_value(self.data)
        return params

    _param_pattern = re.compile(r"\$\{(\w+)\}")

    @staticmethod
    def _extract_params_from_text(text: str) -> Set[str]:
        """Given a string, return all  ${param}  placeholders it contains."""
        if not text:
            return set()
        return set(PromptStructure._param_pattern.findall(text))

    def get_parameters_for_leaf(
        self, leaf: str, model: Optional[str] = None
    ) -> Set[str]:
        """Return the parameters used **only** in the requested leaf/section."""
        content = self.get_leaf_content(leaf, model)
        return self._extract_params_from_text(content)
    
    def get_leaf_content(self, leaf: str, model: Optional[str] = None) -> Optional[str]:
        """Get content for a specific leaf, with optional model override."""
        # Start with defaults
        content = self._get_from_defaults(leaf)
        
        # Apply model override if specified and exists
        if model and content is not None:
            model_override = self._get_from_model(leaf, model)
            if model_override is not None:
                content = model_override
        
        return content
    
    def _get_from_defaults(self, leaf: str) -> Optional[str]:
        defaults = self.data.get('defaults', {})
        if not isinstance(defaults, dict):
            return None
        
        # Check sections first
        if 'sections' in defaults and isinstance(defaults['sections'], dict):
            if leaf in defaults['sections'] and isinstance(defaults['sections'][leaf], str):
                return defaults['sections'][leaf]
        
        # Check direct access
        if leaf in defaults and isinstance(defaults[leaf], str):
            return defaults[leaf]
        
        return None
    
    def _get_from_model(self, leaf: str, model: str) -> Optional[str]:
        models = self.data.get('models', {})
        if not isinstance(models, dict) or model not in models:
            return None
        
        model_data = models[model]
        if not isinstance(model_data, dict):
            return None
        
        # Check sections first
        if 'sections' in model_data and isinstance(model_data['sections'], dict):
            if leaf in model_data['sections'] and isinstance(model_data['sections'][leaf], str):
                return model_data['sections'][leaf]
        
        # Check direct access
        if leaf in model_data and isinstance(model_data[leaf], str):
            return model_data[leaf]
        
        return None
    
    def set_leaf_content(self, leaf: str, content: str, model: Optional[str] = None) -> bool:
        """Set content for a leaf, creating structure as needed."""
        try:
            if model:
                return self._set_in_model(leaf, content, model)
            else:
                return self._set_in_defaults(leaf, content)
        except Exception:
            return False
    
    def _set_in_defaults(self, leaf: str, content: str) -> bool:
        if 'defaults' not in self.data:
            self.data['defaults'] = {}
        
        defaults = self.data['defaults']
        if not isinstance(defaults, dict):
            self.data['defaults'] = defaults = {}
        
        if 'sections' in defaults and isinstance(defaults['sections'], dict):
            # If sections exists, check if this leaf should go there
            if leaf in defaults['sections']:
                # Leaf already exists in sections, update it there
                defaults['sections'][leaf] = content
                # Remove any duplicate direct key
                if leaf in defaults and leaf != 'sections':
                    del defaults[leaf]
            elif leaf in defaults and leaf != 'sections':
                # Leaf exists as direct key, keep it there
                defaults[leaf] = content
            else:
                # New leaf - put it in sections since sections structure exists
                defaults['sections'][leaf] = content
        else:
            # No sections structure, place directly
            if leaf == 'template':
                # Template always goes direct
                defaults[leaf] = content
            else:
                if len(defaults) > 1 or (len(defaults) == 1 and 'template' not in defaults):
                    # Multiple items or no template - use sections
                    if 'sections' not in defaults:
                        defaults['sections'] = {}
                    defaults['sections'][leaf] = content
                else:
                    # Simple structure - place directly
                    defaults[leaf] = content
        
        return True
    
    def _set_in_model(self, leaf: str, content: str, model: str) -> bool:
        if 'models' not in self.data:
            self.data['models'] = {}
        
        if model not in self.data['models']:
            self.data['models'][model] = {}
        
        model_data = self.data['models'][model]
        if not isinstance(model_data, dict):
            self.data['models'][model] = model_data = {}
        
        # Mirror the exact structure from defaults
        defaults = self.data.get('defaults', {})
        if isinstance(defaults, dict):
            if 'sections' in defaults and isinstance(defaults['sections'], dict) and leaf in defaults['sections']:
                # Leaf exists in defaults.sections, mirror that structure
                if 'sections' not in model_data:
                    model_data['sections'] = {}
                model_data['sections'][leaf] = content
            elif leaf in defaults:
                # Leaf exists directly in defaults, mirror that
                model_data[leaf] = content
            else:
                # New leaf, follow defaults structure preference
                if 'sections' in defaults:
                    if 'sections' not in model_data:
                        model_data['sections'] = {}
                    model_data['sections'][leaf] = content
                else:
                    model_data[leaf] = content
        else:
            # No defaults structure to mirror, use direct placement
            model_data[leaf] = content
        
        return True
    
class PromptFileManager:
    """Handles file operations and maintains base/user file relationships."""
    
    def __init__(self, base_root: Path, user_root: Path):
        self.base_root = Path(base_root)
        self.user_root = Path(user_root)
        self.user_root.mkdir(parents=True, exist_ok=True)
    
    def load_prompt_file(self, rel_path: Path) -> Tuple[Optional[PromptStructure], Optional[PromptStructure], List[str]]:
        """Load both base and user versions of a prompt file. Returns (base, user, errors)."""
        base_path = self.base_root / rel_path
        user_path = self.user_root / rel_path
        
        base_structure = None
        user_structure = None
        errors = []
        
        print(f"DEBUG: Attempting to load base file: {base_path}")  # Debug
        print(f"DEBUG: Base file exists: {base_path.exists()}")  # Debug
        
        # Load base file
        if base_path.exists():
            try:
                base_data = load_yaml_file(base_path)
                print(f"DEBUG: Base YAML data: {base_data}")  # Debug
                if base_data:
                    base_structure = PromptStructure(base_data)
                    print(f"DEBUG: Base structure created with {len(base_structure.available_leaves)} leaves")  # Debug
                else:
                    errors.append(f"Base file is empty: {base_path}")
            except Exception as e:
                print(f"DEBUG: Exception loading base file: {e}")  # Debug
                errors.append(f"Error loading base file: {e}")
        else:
            errors.append(f"Base file not found: {base_path}")
        
        # Load user file if it exists
        if user_path.exists():
            try:
                user_data = load_yaml_file(user_path)
                if user_data:
                    user_structure = PromptStructure(user_data)
            except Exception as e:
                errors.append(f"Error loading user file: {e}")
        
        return base_structure, user_structure, errors
    
    def create_user_override(self, rel_path: Path, base_structure: PromptStructure) -> Tuple[Optional[PromptStructure], Optional[str]]:
        """Create a user override file by copying from base. Returns (structure, error)."""
        try:
            user_path = self.user_root / rel_path
            user_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Start with a copy of base data
            user_data = base_structure.data.copy()
            
            # Save to user file
            save_yaml_file(user_path, user_data)
            
            return PromptStructure(user_data), None
        except Exception as e:
            return None, str(e)
    
    def save_user_file(self, rel_path: Path, user_structure: PromptStructure) -> Optional[str]:
        """Save user file. Returns error message if failed, None if successful."""
        try:
            user_path = self.user_root / rel_path
            user_path.parent.mkdir(parents=True, exist_ok=True)
            save_yaml_file(user_path, user_structure.data)
            return None
        except Exception as e:
            return str(e)
        
class PromptEditorScreen(Vertical):
    """Simplified prompt editor with better error handling and validation."""
    
    TITLE = "YAML Prompt Editor"
    
    BINDINGS = [
        Binding("ctrl+s", "save_changes", "Save Changes", show=True),
        Binding("ctrl+c", "copy_base_to_editor", "Copy Base", show=True),
    ]
    
    # Simplified reactive state
    selected_file: reactive[Optional[Path]] = reactive(None)
    selected_leaf: reactive[Optional[str]] = reactive(None)
    edit_mode: reactive[str] = reactive("view_base")  # view_base, edit_user_default, edit_model_override
    selected_model: reactive[Optional[str]] = reactive(None)
    has_unsaved_changes: reactive[bool] = reactive(False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_manager = PromptFileManager(
            base_root=Path(PROMPT_ROOT),
            user_root=self._get_user_prompt_root()
        )
        self.base_structure: Optional[PromptStructure] = None
        self.user_structure: Optional[PromptStructure] = None
        self.available_models: List[Tuple[str, str]] = []
        self._current_leaf_options: List[str] = []
    
    def _get_user_prompt_root(self) -> Path:
        """Get user prompt directory, creating if needed."""
        import platform
        import os
        
        if platform.system() == "Windows":
            user_root = Path(os.getenv("APPDATA", Path.home())) / "hephia" / "prompts"
        else:
            user_root = Path.home() / ".config" / "hephia" / "prompts"
        
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root
    

    def _clean_text_content(self, text: str) -> str:
        """Clean up text content from TextArea to preserve proper formatting."""
        if not text or text.strip() in ["Select a mode and create override to edit...", 
                                        "Enter your custom default content here...",
                                        "Enter content for selected model here..."]:
            return ""
        
        # Remove any trailing whitespace from each line but preserve structure
        lines = text.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)

    def compose(self) -> ComposeResult:
        yield Label("YAML Prompt Editor", classes="screen-title")
        
        with Horizontal(id="main_layout_prompts"):
            # Left panel - navigation
            with Vertical(id="nav_context_panel", classes="panel"):
                yield Label("Base Prompt Files")
                yield Tree("Prompts", id="prompt_file_tree", classes="navigation-tree")
                
                yield Static("---")
                yield Label("Select Section:")
                yield Select([], prompt="Choose section...", id="leaf_selector", disabled=True)
                
                yield Label("Edit Mode:")
                yield Select([
                    ("View Base (Read-Only)", "view_base"),
                    ("Edit User Default", "edit_user_default"),
                    ("Edit Model Override", "edit_model_override"),
                ], value="view_base", id="mode_selector", disabled=True)
                
                yield Label("Model (for overrides):")
                yield Select([], prompt="Choose model...", id="model_selector", disabled=True)

                yield Static("---")
                yield Static("No parameters detected", id="parameter_display", classes="parameter-display")
        
            # Right panel - editing
            with Vertical(id="editing_panel", classes="panel"):
                with Horizontal(classes="button-bar"):
                    yield Button("Save Changes", variant="primary", id="save_btn", disabled=True)
                    yield Button("Copy Base → Editor", id="copy_btn", disabled=True)
                    yield Button("Create User Override", id="create_override_btn", disabled=True)
                
                with Horizontal(id="editor_comparison_layout"):
                    # Base content (read-only)
                    with VerticalScroll(classes="editor-pane"):
                        yield Label("Base Content (Read-Only)")
                        yield TextArea(
                            "", id="base_content", read_only=True,
                            language="yaml", theme="vscode_dark", classes="read-only-editor"
                        )
                    
                    # Editable content
                    with VerticalScroll(classes="editor-pane"):
                        yield Label("Your Content")
                        yield TextArea(
                            "Select a prompt file to begin...", id="user_content",
                            language="yaml", theme="vscode_dark"
                        )
        
        yield Static("Ready", id="status_message", classes="status-message")

    def on_mount(self) -> None:
        """Initialize the editor."""
        self._populate_file_tree()
        self._load_available_models()
        self._update_ui_state()

    def _populate_file_tree(self) -> None:
        """Build the file tree from base prompt directory."""
        tree = self.query_one("#prompt_file_tree", Tree)
        tree.clear()
        tree.root.label = "Base Prompts"
        
        def add_tree_nodes(directory: Path, parent_node: TreeNode) -> None:
            try:
                for item in sorted(directory.iterdir()):
                    if item.is_dir():
                        dir_node = parent_node.add(item.name, data=item, allow_expand=True)
                        add_tree_nodes(item, dir_node)
                    elif item.suffix.lower() in ('.yaml', '.yml'):
                        rel_path = item.relative_to(self.file_manager.base_root)
                        parent_node.add_leaf(item.name, data=rel_path)
            except (OSError, PermissionError):
                # Skip directories we can't read
                pass
        
        base_root = self.file_manager.base_root
        if base_root.exists():
            add_tree_nodes(base_root, tree.root)
        
        tree.root.expand()

    def _load_available_models(self) -> None:
        """Load available models for override selection."""
        try:
            self.available_models = get_all_available_model_names()
            model_select = self.query_one("#model_selector", Select)
            model_select.set_options(self.available_models)
        except Exception as e:
            self._update_status(f"Error loading models: {e}", "error")

    def _update_status(self, message: str, msg_type: str = "info") -> None:
        """Update status message with appropriate styling."""
        status = self.query_one("#status_message", Static)
        if msg_type == "error":
            status.update(f"[red]Error: {message}[/red]")
        elif msg_type == "success":
            status.update(f"[green]✓ {message}[/green]")
        elif msg_type == "warning":
            status.update(f"[yellow]⚠ {message}[/yellow]")
        else:
            status.update(message)

    def _update_ui_state(self) -> None:
        """Update UI elements based on current state."""
        
        # Get current content
        base_content = ""
        user_content = ""
        parameters: Set[str] = set()
        available_leaves = []
        can_edit = False
        content_placeholder = "Select a section and mode to begin editing..."
        
        if self.base_structure:
            available_leaves = self.base_structure.available_leaves
            if self.selected_leaf:
                if self.base_structure:
                    parameters |= self.base_structure.get_parameters_for_leaf(
                        self.selected_leaf
                    )

                if self.user_structure:
                    if (
                        self.edit_mode == "edit_model_override"
                        and self.selected_model
                    ):
                        parameters |= self.user_structure.get_parameters_for_leaf(
                            self.selected_leaf, self.selected_model
                        )
                    else:
                        parameters |= self.user_structure.get_parameters_for_leaf(
                            self.selected_leaf
                        )
                
                base_content = self.base_structure.get_leaf_content(self.selected_leaf) or ""
                
                if self.edit_mode == "view_base":
                    user_content = base_content
                    can_edit = False
                    content_placeholder = "Read-only view of base content"
                elif self.edit_mode == "edit_user_default":
                    if self.user_structure:
                        user_content = self.user_structure.get_leaf_content(self.selected_leaf) or ""
                    else:
                        user_content = "" 
                    content_placeholder = "Enter your custom default content here..."
                    can_edit = True
                elif self.edit_mode == "edit_model_override" and self.selected_model:
                    if self.user_structure:
                        user_content = self.user_structure.get_leaf_content(self.selected_leaf, self.selected_model) or ""
                    else:
                        user_content = ""
                    content_placeholder = f"Enter content for {self.selected_model or 'selected model'} here..."
                    can_edit = True
            else:
                parameters = self.base_structure.parameters
        
        # Update text areas
        self.query_one("#base_content", TextArea).text = base_content
        user_text_area = self.query_one("#user_content", TextArea)
        
        # Use actual content if it exists, otherwise use placeholder
        if user_content:
            user_text_area.text = user_content
        else:
            user_text_area.text = content_placeholder
        
        user_text_area.read_only = not can_edit
        
        # Update leaf selector - ONLY if options changed
        leaf_select = self.query_one("#leaf_selector", Select)
        if available_leaves != self._current_leaf_options:
            self._current_leaf_options = available_leaves.copy()
            if available_leaves:
                leaf_select.set_options([(leaf, leaf) for leaf in available_leaves])
                leaf_select.disabled = False
            else:
                leaf_select.set_options([])
                leaf_select.disabled = True
        
        # Update mode selector
        mode_select = self.query_one("#mode_selector", Select)
        mode_select.disabled = not self.selected_file
        
        # Update model selector
        model_select = self.query_one("#model_selector", Select)
        model_select.disabled = self.edit_mode != "edit_model_override"
        
        # Update buttons
        self.query_one("#save_btn", Button).disabled = not can_edit or not self.has_unsaved_changes
        self.query_one("#copy_btn", Button).disabled = not can_edit or not base_content
        self.query_one("#create_override_btn", Button).disabled = not self.selected_file
        
        # Update parameters display
        param_display = self.query_one("#parameter_display", Static)
        if parameters:
            param_list = ", ".join(f"`${{{p}}}`" for p in sorted(parameters))
            param_display.update(f"Parameters: {param_list}")
        else:
            param_display.update("No parameters detected")

    # Event handlers
    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle file selection from tree."""
        print(f"DEBUG: Tree node selected: {event.node}")  # Debug
        print(f"DEBUG: Node data: {event.node.data if event.node else None}")  # Debug
        print(f"DEBUG: Allow expand: {event.node.allow_expand if event.node else None}")  # Debug
        
        if event.node and event.node.data and not event.node.allow_expand:
            print(f"DEBUG: Setting selected_file to: {event.node.data}")  # Debug
            self.selected_file = event.node.data
        else:
            print("DEBUG: Clearing selected_file")  # Debug
            self.selected_file = None

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id == "leaf_selector":
            self.selected_leaf = str(event.value) if event.value != Select.BLANK else None
        elif event.select.id == "mode_selector":
            self.edit_mode = str(event.value)
        elif event.select.id == "model_selector":
            self.selected_model = str(event.value) if event.value != Select.BLANK else None

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes."""
        if event.text_area.id == "user_content" and not event.text_area.read_only:
            self.has_unsaved_changes = True

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_btn":
            await self.action_save_changes()
        elif event.button.id == "copy_btn":
            await self.action_copy_base_to_editor()
        elif event.button.id == "create_override_btn":
            await self.action_create_user_override()

    # Watch methods for reactive properties
    def watch_selected_file(self, old_file: Optional[Path], new_file: Optional[Path]) -> None:
        if new_file:
            self._load_file(new_file)
        else:
            self._clear_file_data()
        self._update_ui_state()

    def watch_selected_leaf(self, old_leaf: Optional[str], new_leaf: Optional[str]) -> None:
        self.has_unsaved_changes = False
        self._update_ui_state()

    def watch_edit_mode(self, old_mode: str, new_mode: str) -> None:
        self.has_unsaved_changes = False
        if new_mode != "edit_model_override":
            self.selected_model = None
        self._update_ui_state()

    def watch_edit_mode(self, old_mode: str, new_mode: str) -> None:
        # Clear unsaved changes when switching modes
        self.has_unsaved_changes = False
        if new_mode != "edit_model_override":
            self.selected_model = None
        
        # Clear the editor content when switching to edit mode to start fresh
        if new_mode in ["edit_user_default", "edit_model_override"]:
            user_text_area = self.query_one("#user_content", TextArea)
            if not user_text_area.read_only:
                # Only clear if we don't have actual saved content to load
                current_content = self._get_saved_content_for_current_context()
                if not current_content:
                    user_text_area.text = ""
        
        self._update_ui_state()

    def _get_saved_content_for_current_context(self) -> str:
        """Get any saved content that should be loaded for current context."""
        if not self.user_structure or not self.selected_leaf:
            return ""
        
        if self.edit_mode == "edit_user_default":
            return self.user_structure.get_leaf_content(self.selected_leaf) or ""
        elif self.edit_mode == "edit_model_override" and self.selected_model:
            return self.user_structure.get_leaf_content(self.selected_leaf, self.selected_model) or ""
        
        return ""

    def watch_has_unsaved_changes(self, old_changes: bool, new_changes: bool) -> None:
        save_btn = self.query_one("#save_btn", Button)
        if new_changes:
            save_btn.label = "Save Changes*"
        else:
            save_btn.label = "Save Changes"
        self._update_ui_state()

    # Helper methods
    def _load_file(self, rel_path: Path) -> None:
        """Load base and user structures for the selected file."""
        try:
            print(f"DEBUG: Loading file: {rel_path}")  # Debug line
            self.base_structure, self.user_structure, errors = self.file_manager.load_prompt_file(rel_path)
            
            print(f"DEBUG: Base structure loaded: {self.base_structure is not None}")  # Debug
            if self.base_structure:
                print(f"DEBUG: Available leaves: {self.base_structure.available_leaves}")  # Debug
            
            if errors:
                print(f"DEBUG: Errors: {errors}")  # Debug
                self._update_status("; ".join(errors), "error")
            elif self.base_structure:
                num_leaves = len(self.base_structure.available_leaves)
                self._update_status(f"Loaded: {rel_path.name} ({num_leaves} sections)")
            else:
                self._update_status(f"Could not load: {rel_path.name}", "error")
            
            # Reset selections
            self.selected_leaf = None
            self.selected_model = None
            self.has_unsaved_changes = False
            
        except Exception as e:
            print(f"DEBUG: Exception in _load_file: {e}")  # Debug
            self._update_status(f"Error loading file: {e}", "error")

    def _clear_file_data(self) -> None:
        """Clear all file-related data."""
        self.base_structure = None
        self.user_structure = None
        self.selected_leaf = None
        self.selected_model = None
        self.has_unsaved_changes = False
        self._current_leaf_options = []

    # Actions
    async def action_save_changes(self) -> None:
        """Save current changes to user override file."""
        if not self.selected_file or not self.selected_leaf or not self.base_structure:
            self._update_status("Cannot save: no file/leaf selected", "error")
            return
        
        # Ensure we have a user structure
        if not self.user_structure:
            await self.action_create_user_override()
            if not self.user_structure:
                return  # Creation failed
        
        # Get and clean content from editor
        raw_content = self.query_one("#user_content", TextArea).text
        user_content = self._clean_text_content(raw_content)
        
        # If content is empty after cleaning, don't save empty strings
        if not user_content:
            self._update_status("Cannot save empty content", "error")
            return
        
        # Validate parameters
        user_params = set(re.findall(r'\$\{(\w+)\}', user_content))
        base_params = self.base_structure.parameters
        invalid_params = user_params - base_params
        
        if invalid_params:
            invalid_list = ", ".join(sorted(invalid_params))
            self._update_status(f"Invalid parameters detected: {invalid_list}", "error")
            return
        
        # Set content in user structure
        model = self.selected_model if self.edit_mode == "edit_model_override" else None
        success = self.user_structure.set_leaf_content(self.selected_leaf, user_content, model)
        
        if not success:
            self._update_status("Failed to update content structure", "error")
            return
        
        # Save to file
        error = self.file_manager.save_user_file(self.selected_file, self.user_structure)
        if error:
            self._update_status(f"Save failed: {error}", "error")
        else:
            self.has_unsaved_changes = False
            self._update_status("Changes saved successfully", "success")

    async def action_copy_base_to_editor(self) -> None:
        """Copy base content to editor."""
        if not self.base_structure or not self.selected_leaf:
            return
        
        base_content = self.base_structure.get_leaf_content(self.selected_leaf) or ""
        self.query_one("#user_content", TextArea).text = base_content
        self.has_unsaved_changes = True
        self._update_status("Base content copied to editor")

    async def action_create_user_override(self) -> None:
        """Create user override file if it doesn't exist."""
        if not self.selected_file or not self.base_structure:
            self._update_status("Cannot create override: no base file selected", "error")
            return
        
        if self.user_structure:
            self._update_status("User override already exists")
            return
        
        # Check if user file exists
        user_path = self.file_manager.user_root / self.selected_file
        if user_path.exists():
            async def handle_overwrite(confirmed: bool) -> None:
                if confirmed:
                    await self._do_create_override()
            
            self.app.push_screen(
                ConfirmationDialog(
                    prompt=f"User override file exists: {user_path.name}\nOverwrite with base content?",
                    confirm_button_label="Overwrite",
                    confirm_button_variant="warning"
                ),
                handle_overwrite
            )
        else:
            await self._do_create_override()

    async def _do_create_override(self) -> None:
        """Actually create the user override file."""
        if not self.selected_file or not self.base_structure:
            return
        
        self.user_structure, error = self.file_manager.create_user_override(
            self.selected_file, self.base_structure
        )
        
        if error:
            self._update_status(f"Failed to create override: {error}", "error")
        else:
            self._update_status("User override created successfully", "success")
            # Switch to edit mode
            self.edit_mode = "edit_user_default"
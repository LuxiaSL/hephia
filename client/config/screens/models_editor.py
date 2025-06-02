# client/config/screens/models_editor.py
"""
Textual screen for editing custom LLM model definitions (models.json).
"""
from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.css.query import DOMQuery
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DataTable, Label, Input, Select, Static, Checkbox,
    Markdown
)
from textual.message import Message

from typing import Dict, Optional, Any, Tuple, List

# Assuming execution from project root for these imports
from config import ModelConfig as MainModelConfig, ProviderType as MainProviderType
from ..dialogs import ConfirmationDialog
from ..utils import load_models_json, save_models_json, get_models_json_path, MODELS_JSON_PATH
from ..models import ModelConfig as TuiModelConfig

# --- Helper classes and messages ---

class ModelFormState(Message):
    """Message to pass model data to/from the form."""
    def __init__(self, model_name: Optional[str], model_config: Optional[TuiModelConfig], is_new: bool):
        self.model_name = model_name
        self.model_config = model_config
        self.is_new = is_new
        super().__init__()

class ModelFormDialog(ModalScreen[Optional[ModelFormState]]):
    """A modal dialog for adding or editing a model configuration."""

    # External data passed in
    model_name_to_edit: Optional[str] = None
    model_config_to_edit: Optional[TuiModelConfig] = None
    is_new_model: bool = True
    
    # Store all existing model names to check for duplicates if adding/renaming
    existing_model_names: List[str] = []

    BINDINGS = [
        Binding("escape", "cancel_edit", "Cancel", show=False),
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_config: Optional[TuiModelConfig] = None,
        is_new: bool = True,
        all_model_names: Optional[List[str]] = None, # Pass existing names
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None
    ) -> None:
        super().__init__(name, id, classes)
        self.model_name_to_edit = model_name
        self.model_config_to_edit = model_config
        self.is_new_model = is_new
        self.existing_model_names = all_model_names or []


    def compose(self) -> ComposeResult:
        title = "Add New Custom Model" if self.is_new_model else f"Edit Model: {self.model_name_to_edit}"
        mc = self.model_config_to_edit

        with Vertical(classes="dialog modal-dialog-compact"):
            yield Label(title, classes="dialog-title")
            
            # Two-column layout for better space usage
            with Horizontal(classes="dialog-form-layout"):
                # Left column - Basic info
                with Vertical(classes="dialog-column"):
                    yield Label("Basic Information", classes="dialog-section-header")
                    
                    # Model name field
                    with Horizontal(classes="compact-field"):
                        yield Label("Name:", classes="field-label-inline")
                        
                        if self.is_new_model:
                            yield Input(
                                value=self.model_name_to_edit or "",
                                placeholder="my-custom-model",
                                id="model_name_input",
                                classes="field-input-compact"
                            )
                        else:
                            yield Static(self.model_name_to_edit or "N/A", classes="field-value-static")

                    # Provider field
                    with Horizontal(classes="compact-field"):
                        yield Label("Provider:", classes="field-label-inline")
                        
                        # Create provider options as (display_name, enum_member) pairs
                        provider_options = [(pt.value, pt) for pt in MainProviderType]
                        current_provider = mc.provider if mc else Select.BLANK
                        
                        yield Select(
                            options=provider_options,
                            value=current_provider,
                            id="provider_select",
                            allow_blank=True,
                            classes="field-select-compact"
                        )

                    # Model ID field
                    with Horizontal(classes="compact-field"):
                        yield Label("Model ID:", classes="field-label-inline")
                        yield Input(
                            value=mc.model_id if mc else "", 
                            placeholder="gpt-4-turbo", 
                            id="model_id_input",
                            classes="field-input-compact"
                        )
                
                # Right column - Configuration
                with Vertical(classes="dialog-column"):
                    yield Label("Configuration", classes="dialog-section-header")
                    
                    # Max tokens field
                    with Horizontal(classes="compact-field"):
                        yield Label("Max Tokens:", classes="field-label-inline")
                        yield Input(
                            value=str(mc.max_tokens) if mc else "250", 
                            id="max_tokens_input", 
                            classes="field-input-compact"
                        )

                    # Temperature field
                    with Horizontal(classes="compact-field"):
                        yield Label("Temperature:", classes="field-label-inline")
                        yield Input(
                            value=str(mc.temperature) if mc else "0.7",
                            id="temperature_input", 
                            classes="field-input-compact"
                        )

                    # API key var field
                    with Horizontal(classes="compact-field"):
                        yield Label("API Key Var:", classes="field-label-inline")
                        yield Input(
                            value=mc.env_var if mc else "",
                            placeholder="CUSTOM_API_KEY", 
                            id="env_var_input",
                            classes="field-input-compact"
                        )
            
            # Description spans full width
            yield Label("Description:")
            yield Input(
                value=mc.description if mc else "", 
                placeholder="Brief description of this model",
                id="description_input", 
                classes="field-input-full"
            )
            
            yield Static("", id="validation_error_models", classes="error-message-compact")

            with Horizontal(classes="dialog-buttons-compact"):
                yield Button("Save Model", variant="primary", id="save_model_button")
                yield Button("Cancel", id="cancel_model_button")

    def _get_input_value(self, input_id: str, default: Any = "") -> str:
        try:
            return self.query_one(f"#{input_id}", Input).value
        except Exception:
            return str(default) # Should not happen if IDs are correct

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_model_button":
            error_display = self.query_one("#validation_error_models", Static)
            error_display.update("") # Clear previous errors

            try:
                model_name_str: Optional[str]
                if self.is_new_model:
                    model_name_str = self.query_one("#model_name_input", Input).value.strip()
                    if not model_name_str:
                        error_display.update("[b red]Model Name cannot be empty.[/b red]")
                        return
                    if model_name_str in self.existing_model_names:
                        error_display.update(f"[b red]Model Name '{model_name_str}' already exists.[/b red]")
                        return
                else:
                    model_name_str = self.model_name_to_edit
                
                if not model_name_str: # Should not happen if logic is correct
                    error_display.update("[b red]Internal Error: Model name is missing.[/b red]")
                    return

                # Max tokens and temperature need to be converted to int/float
                try:
                    max_tokens = int(self._get_input_value("max_tokens_input", "250"))
                    if max_tokens <= 0:
                        error_display.update("[b red]Max Tokens must be a positive integer.[/b red]")
                        return
                except ValueError:
                    error_display.update("[b red]Max Tokens must be an integer.[/b red]")
                    return
                
                try:
                    temperature = float(self._get_input_value("temperature_input", "0.7"))
                    if not (0.0 <= temperature <= 2.0):
                        error_display.update("[b red]Temperature must be between 0.0 and 2.0.[/b red]")
                        return
                except ValueError:
                    error_display.update("[b red]Temperature must be a number (e.g., 0.7).[/b red]")
                    return

                # Provider is an enum member from Select
                provider_value = self.query_one("#provider_select", Select).value
                if provider_value == Select.BLANK or provider_value is None:
                    error_display.update("[b red]Provider must be selected.[/b red]")
                    return
                
                # Ensure provider_value is an enum member
                if isinstance(provider_value, str):
                    try:
                        provider_value = MainProviderType(provider_value)
                    except ValueError:
                        error_display.update(f"[b red]Invalid provider: {provider_value}[/b red]")
                        return
                    
                # Validate model_id is not empty
                model_id = self._get_input_value("model_id_input").strip()
                if not model_id:
                    error_display.update("[b red]Model ID cannot be empty.[/b red]")
                    return

                model_data = TuiModelConfig(
                    provider=provider_value,
                    model_id=model_id,
                    env_var=self._get_input_value("env_var_input").strip() or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    description=self._get_input_value("description_input").strip()
                )

                self.dismiss(ModelFormState(model_name_str, model_data, self.is_new_model))

            except ValidationError as e: # Pydantic validation error
                error_display.update(f"[b red]Validation Error:[/b red]\n{e}")
            except Exception as e: # Other unexpected errors
                error_display.update(f"[b red]An unexpected error occurred:[/b red]\n{e}")
        
        elif event.button.id == "cancel_model_button":
            self.dismiss(None)
            
    def action_cancel_edit(self) -> None:
        """Called when escape is pressed."""
        self.dismiss(None)


class ModelsEditorScreen(Vertical):
    """Manages custom LLM model definitions from models.json."""

    TITLE = "Custom Models (models.json)"
    BINDINGS = [
        Binding("a", "add_model", "Add New Model", show=True),
        Binding("e", "edit_selected_model", "Edit Selected", show=True),
        Binding("d", "delete_selected_model", "Delete Selected", show=True),
        Binding("ctrl+s", "save_all_models", "Save All to Disk", show=True),
        Binding("r", "reload_models_from_disk", "Reload from Disk", show=True),
    ]

    # Store custom models: Dict[model_name, ModelConfig_from_main_project]
    custom_models: reactive[Dict[str, MainModelConfig]] = reactive(dict)
    # Track if changes have been made
    has_unsaved_changes: reactive[bool] = reactive(False)
    # Track the currently selected model name
    selected_model_name: reactive[Optional[str]] = reactive(None)

    def compose(self) -> ComposeResult:
        yield Label("Custom LLM Models (models.json)", classes="screen-title")
        yield Markdown(f"Models are loaded from/saved to: `{str(get_models_json_path())}`")
        with Horizontal(classes="button-bar"):
            yield Button("Add New Model", variant="success", id="add_model_btn")
            yield Button("Edit Selected", id="edit_model_btn", disabled=True)
            yield Button("Delete Selected", variant="error", id="delete_model_btn", disabled=True)
        
        yield DataTable(id="models_table", cursor_type="row")
        
        with Horizontal(classes="button-bar"):
            yield Button("Save All Changes to Disk", variant="primary", id="save_all_btn")
            yield Button("Reload from Disk", id="reload_all_btn")
        yield Static(id="models_status_message", classes="status-message")

    def on_mount(self) -> None:
        """Load data and configure table when screen is mounted."""
        table = self.query_one(DataTable)
        table.add_columns("Model Name", "Provider", "Model ID", "Max Tokens", "Temperature", "Description")
        self.load_and_display_models()

    def _update_table(self) -> None:
        """Clears and repopulates the DataTable from self.custom_models."""
        table = self.query_one(DataTable)
        table.clear()
        self.selected_model_name = None  # Clear selection when table updates

        sorted_model_names = sorted(self.custom_models.keys())

        for model_name in sorted_model_names:
            mc = self.custom_models[model_name]
            
            # Safe provider display - handle both enum members and strings
            provider_display = "N/A"
            if mc.provider:
                if hasattr(mc.provider, 'value'):
                    provider_display = mc.provider.value
                else:
                    provider_display = str(mc.provider)
            
            # Use model_name as the row key directly
            table.add_row(
                model_name,
                provider_display,
                mc.model_id,
                str(mc.max_tokens),
                str(mc.temperature),
                mc.description,
                key=model_name
            )
        
        self._update_action_button_states()
        unsaved_indicator = "[b yellow]Unsaved changes.[/b yellow]" if self.has_unsaved_changes else ""
        self.query_one("#models_status_message", Static).update(
            f"{len(self.custom_models)} custom models loaded. {unsaved_indicator}"
        )

    def load_and_display_models(self) -> None:
        """Loads models from models.json and displays them."""
        try:
            loaded_data = load_models_json()
        except RuntimeError as e:
            # If file doesn't exist or is corrupted, start with empty models
            self.custom_models = {}
            self.has_unsaved_changes = False
            self._update_table()
            self.query_one("#models_status_message", Static).update(f"[b yellow]Warning: {e}[/b yellow]")
            return
        
        models_temp_dict: Dict[str, MainModelConfig] = {}
        parse_errors = []

        for name, data_dict in loaded_data.items():
            try:
                # Ensure provider is converted to enum member
                provider_str = data_dict.get("provider")
                if provider_str:
                    if isinstance(provider_str, str):
                        data_dict["provider"] = MainProviderType(provider_str)
                
                # Create MainModelConfig with proper validation
                models_temp_dict[name] = MainModelConfig(
                    provider=data_dict["provider"],
                    model_id=data_dict["model_id"],
                    env_var=data_dict.get("env_var"),
                    max_tokens=int(data_dict.get("max_tokens", 250)),
                    temperature=float(data_dict.get("temperature", 0.7)),
                    description=data_dict.get("description", "")
                )
            except Exception as e:
                parse_errors.append(f"Error parsing model '{name}': {e}")
        
        self.custom_models = models_temp_dict
        self.has_unsaved_changes = False
        self._update_table()

        status_message = f"Loaded {len(models_temp_dict)} models successfully."
        if parse_errors:
            error_text = f"{status_message} [b red]Warnings:[/b red]\n" + "\n".join(parse_errors[:3])
            if len(parse_errors) > 3:
                error_text += f"\n... and {len(parse_errors) - 3} more errors"
            self.query_one("#models_status_message", Static).update(error_text)
            self.app.bell()
        else:
            self.query_one("#models_status_message", Static).update(status_message)

    def _update_action_button_states(self) -> None:
        """Enable/disable edit/delete buttons based on table selection."""
        has_selection = self.selected_model_name is not None
        
        edit_button: DOMQuery[Button] = self.query_one("#edit_model_btn", Button)
        delete_button: DOMQuery[Button] = self.query_one("#delete_model_btn", Button)
        
        edit_button.disabled = not has_selection
        delete_button.disabled = not has_selection

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection - this fires when a row is clicked/selected."""
        try:
            table = self.query_one(DataTable)
            if event.cursor_row is not None and event.cursor_row < table.row_count:
                # Get the model name from the first column of the selected row
                row_data = table.get_row_at(event.cursor_row)
                self.selected_model_name = str(row_data[0]) if row_data else None
            else:
                self.selected_model_name = None
        except Exception as e:
            print(f"Error in row selection: {e}")
            self.selected_model_name = None
        self._update_action_button_states()

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlighting - this fires when cursor moves over rows."""
        try:
            table = self.query_one(DataTable)
            if event.cursor_row is not None and event.cursor_row < table.row_count:
                row_data = table.get_row_at(event.cursor_row)
                self.selected_model_name = str(row_data[0]) if row_data else None
            else:
                self.selected_model_name = None
        except Exception as e:
            print(f"Error in row highlighting: {e}")
            self.selected_model_name = None
        self._update_action_button_states()

    def _get_selected_model_name(self) -> Optional[str]:
        """Get the currently selected model name."""
        return self.selected_model_name

    def _show_model_form(self, model_name: Optional[str], model_config: Optional[MainModelConfig], is_new: bool) -> None:
        """Helper to show the modal form."""
        tui_mc: Optional[TuiModelConfig] = None
        if model_config:
            try:
                # Ensure provider is an enum member
                provider_enum = model_config.provider
                if isinstance(provider_enum, str):
                    provider_enum = MainProviderType(provider_enum)
                elif not isinstance(provider_enum, MainProviderType):
                    print(f"Warning: Invalid provider type {type(provider_enum)}, converting from {provider_enum}")
                    provider_enum = MainProviderType(str(provider_enum))

                tui_mc = TuiModelConfig(
                    provider=provider_enum,
                    model_id=model_config.model_id,
                    env_var=model_config.env_var,
                    max_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                    description=model_config.description
                )
            except Exception as e:
                self.query_one("#models_status_message", Static).update(f"[b red]Error preparing form: {e}[/b red]")
                self.app.bell()
                return

        self.app.push_screen(
            ModelFormDialog(
                model_name=model_name,
                model_config=tui_mc,
                is_new=is_new,
                all_model_names=list(self.custom_models.keys())
            ),
            self._handle_form_dismiss
        )

    async def _handle_form_dismiss(self, result: Optional[ModelFormState]) -> None:
        """Callback for when the ModelFormDialog is dismissed."""
        if result and result.model_name and result.model_config:
            try:
                # Convert TuiModelConfig back to MainModelConfig for storage
                main_mc = MainModelConfig(
                    provider=result.model_config.provider,
                    model_id=result.model_config.model_id,
                    env_var=result.model_config.env_var,
                    max_tokens=result.model_config.max_tokens,
                    temperature=result.model_config.temperature,
                    description=result.model_config.description
                )
                self.custom_models[result.model_name] = main_mc
                self.has_unsaved_changes = True
                self._update_table()
                action_type = "added" if result.is_new else "updated"
                self.query_one("#models_status_message", Static).update(
                    f"Model '{result.model_name}' {action_type}. [b yellow]Unsaved changes.[/b yellow]"
                )
                
                # Select the newly added/edited model
                self.selected_model_name = result.model_name
                self._update_action_button_states()
                
            except Exception as e:
                 self.query_one("#models_status_message", Static).update(f"[b red]Error processing form data: {e}[/b red]")
                 self.app.bell()

    def action_add_model(self) -> None:
        """Action to add a new model."""
        self._show_model_form(None, None, is_new=True)

    def action_edit_selected_model(self) -> None:
        """Action to edit the selected model."""
        selected_model_name = self._get_selected_model_name()
        if selected_model_name and selected_model_name in self.custom_models:
            model_config = self.custom_models[selected_model_name]
            self._show_model_form(selected_model_name, model_config, is_new=False)
        else:
            # Escape the debug info to avoid markup conflicts
            current_selection = f"Current selection: {repr(selected_model_name)}"
            available_models = f"Available models: {repr(list(self.custom_models.keys()))}"
            self.query_one("#models_status_message", Static).update(
                f"[b red]No model selected or model not found.[/b red]\n{current_selection}\n{available_models}"
            )
            self.app.bell()
            
    def action_delete_selected_model(self) -> None:
        """Action to delete the selected model."""
        selected_model_name = self._get_selected_model_name()
        if selected_model_name and selected_model_name in self.custom_models:
            async def check_confirmation(confirmed: bool) -> None:
                if confirmed:
                    del self.custom_models[selected_model_name]
                    self.has_unsaved_changes = True
                    self.selected_model_name = None  # Clear selection after delete
                    self._update_table()
                    self.query_one("#models_status_message", Static).update(
                        f"Model '{selected_model_name}' deleted. [b yellow]Unsaved changes.[/b yellow]"
                    )
                else:
                    self.query_one("#models_status_message", Static).update("Deletion cancelled.")

            self.app.push_screen(
                ConfirmationDialog(
                    prompt=f"Are you sure you want to delete model '{selected_model_name}'?",
                    confirm_button_variant="error",
                    confirm_button_label="Delete"
                ),
                check_confirmation
            )
        else:
            self.query_one("#models_status_message", Static).update("[b red]No model selected for deletion.[/b red]")
            self.app.bell()

    def action_save_all_models(self) -> None:
        """Saves all custom models to models.json."""
        models_to_save: Dict[str, Any] = {}
        conversion_errors = []
        
        for name, mc in self.custom_models.items():
            try:
                # Convert to dictionary for JSON serialization
                model_dict = {
                    "provider": mc.provider.value if hasattr(mc.provider, 'value') else str(mc.provider),
                    "model_id": mc.model_id,
                    "max_tokens": mc.max_tokens,
                    "temperature": mc.temperature,
                    "description": mc.description
                }
                if mc.env_var:
                    model_dict["env_var"] = mc.env_var
                
                models_to_save[name] = model_dict
            except Exception as e:
                conversion_errors.append(f"Error converting model '{name}': {e}")
        
        if conversion_errors:
            error_text = "[b red]Errors preparing models for save:[/b red]\n" + "\n".join(conversion_errors)
            self.query_one("#models_status_message", Static).update(error_text)
            self.app.bell()
            return
            
        try:
            backup_made = save_models_json(models_to_save)
            backup_msg = f" Backup created ({MODELS_JSON_PATH.name}.bak)." if backup_made else ""
            
            self.has_unsaved_changes = False
            self._update_table()
            self.query_one("#models_status_message", Static).update(
                f"[b green]All models saved!{backup_msg}[/b green] Restart Hephia for changes to take effect."
            )
        except Exception as e:
            self.query_one("#models_status_message", Static).update(f"[b red]Error saving models: {e}[/b red]")
            self.app.bell()

    def action_reload_models_from_disk(self) -> None:
        """Reloads models from disk, discarding any unsaved changes."""
        if self.has_unsaved_changes:
            async def check_confirmation(confirmed: bool) -> None:
                if confirmed:
                    self.load_and_display_models()
                    self.query_one("#models_status_message", Static).update("Models reloaded from disk.")
                else:
                    self.query_one("#models_status_message", Static).update("Reload cancelled.")

            self.app.push_screen(
                ConfirmationDialog(
                    prompt="You have unsaved changes. Are you sure you want to reload from disk?",
                    confirm_button_variant="warning",
                    confirm_button_label="Reload"
                ),
                check_confirmation
            )
        else:
            self.load_and_display_models()
            self.query_one("#models_status_message", Static).update("Models reloaded from disk.")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for actions not covered by bindings."""
        if event.button.id == "add_model_btn":
            self.action_add_model()
        elif event.button.id == "edit_model_btn":
            self.action_edit_selected_model()
        elif event.button.id == "delete_model_btn":
            self.action_delete_selected_model()
        elif event.button.id == "save_all_btn":
            self.action_save_all_models()
        elif event.button.id == "reload_all_btn":
            self.action_reload_models_from_disk()
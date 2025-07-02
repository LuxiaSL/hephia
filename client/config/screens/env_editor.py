# client/config/screens/env_editor.py
"""
Textual screen for editing Hephia's .env file settings.
"""
from textual import on
from textual.app import ComposeResult
from textual.widgets import (
    Label, Button, Input, Checkbox, Static, Select,
    Markdown
)
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.reactive import reactive
from textual.message import Message
from textual.dom import NoMatches

from typing import Dict, Any, Optional, Type, Tuple, List
from pydantic import ValidationError

from ..models import EnvConfigModel # The Pydantic model for .env
from ..utils import load_dotenv_values, save_dotenv_value, get_all_available_model_names, backup_dotenv_file, DOTENV_PATH


class EnvEditorScreen(Vertical):
    """
    Manages the display and editing of .env configuration variables.
    """

    # Store the .env data parsed into our Pydantic model
    env_data: reactive[Optional[EnvConfigModel]] = reactive(None)
    # Store initial raw loaded values to compare for changes
    initial_raw_values: reactive[Dict[str, Optional[str]]] = reactive(dict)

    # Message to indicate save completion or error
    class EnvOpComplete(Message):
        def __init__(self, success: bool, message_text: str) -> None:
            self.success = success
            self.message_text = message_text
            super().__init__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._form_widgets: Dict[str, Any] = {} # To hold references to input widgets

    async def on_mount(self) -> None:
        """Load data when the screen is mounted."""
        await self.load_env_data()

    async def load_env_data(self) -> None:
        """Loads .env values and populates the Pydantic model."""
        raw_values = load_dotenv_values()
        self.initial_raw_values = raw_values.copy()
        
        # Convert string values to appropriate types for Pydantic
        converted_values = {}
        for key, value in raw_values.items():
            if value is None:
                converted_values[key] = None
            elif key in EnvConfigModel.model_fields:
                field_info = EnvConfigModel.model_fields[key]
                field_type = field_info.annotation
                
                # Handle type conversion from string
                if field_type == bool:
                    # Convert string boolean to actual boolean
                    if value.lower() in ('true', '1', 'yes', 'on'):
                        converted_values[key] = True
                    elif value.lower() in ('false', '0', 'no', 'off'):
                        converted_values[key] = False
                    else:
                        converted_values[key] = bool(value)  # Fallback
                elif field_type == float:
                    try:
                        converted_values[key] = float(value)
                    except (ValueError, TypeError):
                        converted_values[key] = field_info.default
                elif field_type == int:
                    try:
                        converted_values[key] = int(float(value))  # Handle "15.0" -> 15
                    except (ValueError, TypeError):
                        converted_values[key] = field_info.default
                else:
                    # String or other types
                    converted_values[key] = value
            else:
                converted_values[key] = value
        
        try:
            # Pydantic will coerce types where possible
            self.env_data = EnvConfigModel(**converted_values)
        except ValidationError as e:
            self.env_data = None
            # TODO: Display validation errors to the user in a more structured way
            self.query_one("#status_message", Static).update(
                f"[b red]Error loading .env:[/b red]\n{e}"
            )
            return
        
        # If env_data was loaded, (re)compose the form
        if self.env_data:
            await self.recompose_form()

    def compose(self) -> ComposeResult:
        """Compose the screen UI with grid layout."""
        with Vertical(id="env_editor_layout"):
            yield Label("Environment Variables (.env)", classes="screen-title")
            with Horizontal(classes="button-bar"):
                yield Button("Save Changes", variant="primary", id="save_env")
                yield Button("Reload from Disk", id="reload_env")
            
            # Main form in horizontal sections
            with Horizontal(id="env_form_main"):
                # Left column - API Keys section
                with Vertical(id="api_keys_section", classes="form-section"):
                    yield Label("🔑 API Keys", classes="section-header")
                    with VerticalScroll(classes="form-scroll"):
                        yield Vertical(id="api_keys_container")
                
                # Right column - Configuration section  
                with Vertical(id="config_section", classes="form-section"):
                    yield Label("⚙️ Configuration", classes="section-header")
                    with VerticalScroll(classes="form-scroll"):
                        # Models subsection
                        yield Label("Models", classes="subsection-header")
                        yield Vertical(id="models_container")
                        
                        # System Settings subsection
                        yield Label("System", classes="subsection-header") 
                        yield Vertical(id="system_container")
                        
                        # Discord subsection
                        yield Label("Discord", classes="subsection-header")
                        yield Vertical(id="discord_container")
                        
                        # Advanced subsection
                        yield Label("Advanced", classes="subsection-header")
                        yield Vertical(id="advanced_container")
            
            yield Static(id="fixed_status_message", classes="fixed-status-message")

    async def recompose_form(self) -> None:
        """Clears and rebuilds the form based on self.env_data using grouped containers."""
        # Clear all containers
        container_ids = [
            "api_keys_container", "models_container", "system_container", 
            "discord_container", "advanced_container"
        ]
        
        for container_id in container_ids:
            container = self.query_one(f"#{container_id}", Vertical)
            await container.remove_children()
        
        self._form_widgets.clear()

        # Add status message to first container
        api_container = self.query_one("#api_keys_container", Vertical)
        status = Static(id="status_message", classes="status-message")
        api_container.mount(status)
        
        # Group fields by category
        api_key_fields = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", 
            "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "PERPLEXITY_API_KEY",
            "OPENPIPE_API_KEY", "CHAPTER2_API_KEY", "DISCORD_BOT_TOKEN"
        ]
        
        model_fields = [
            "COGNITIVE_MODEL", "VALIDATION_MODEL", "SUMMARY_MODEL", "FALLBACK_MODEL"
        ]
        
        system_fields = [
            "EXO_MIN_INTERVAL", "EXO_MAX_TURNS", "HEADLESS", "USE_LOCAL_EMBEDDING"
        ]
        
        discord_fields = [
            "ENABLE_DISCORD", "REPLY_ON_TAG"
        ]
        
        advanced_fields = [
            "LOG_PROMPTS", "ADVANCED_C2_LOGGING", "CHAPTER2_SOCKET_PATH", "CHAPTER2_HTTP_PORT", "LOCAL_INFERENCE_BASE_URL"
        ]
        
        # Build each section
        await self._build_field_group("api_keys_container", api_key_fields)
        await self._build_field_group("models_container", model_fields) 
        await self._build_field_group("system_container", system_fields)
        await self._build_field_group("discord_container", discord_fields)
        await self._build_field_group("advanced_container", advanced_fields)

    async def _build_field_group(self, container_id: str, field_names: List[str]) -> None:
        """Build a group of fields in a container."""
        container = self.query_one(f"#{container_id}", Vertical)
        
        for field_name in field_names:
            if field_name not in EnvConfigModel.model_fields:
                continue
                
            field_info = EnvConfigModel.model_fields[field_name]
            current_value = getattr(self.env_data, field_name, field_info.default) if self.env_data else field_info.default
            description = field_info.description or f"Set {field_name}"
            field_type = field_info.annotation
            
            # Create compact field layout
            field_container = Horizontal(classes="compact-field")
            container.mount(field_container)
            
            # Label (shorter)
            label_text = field_name.replace("_", " ").title()
            if field_name.endswith("_KEY") or field_name.endswith("_TOKEN"):
                label_text = label_text.replace(" Key", "").replace(" Token", "")
            
            label = Label(label_text + ":", classes="field-label-compact")
            field_container.mount(label)
            
            # Widget (takes remaining space)
            widget = self._create_field_widget(field_name, field_type, current_value, description)
            if widget:
                field_container.mount(widget)
                self._form_widgets[field_name] = widget
            
            # Help text below (optional, only for complex fields)
            # if field_name in ["EXO_MIN_INTERVAL"]:
            #     help_text = Static(description, classes="help-text-compact")
            #     container.mount(help_text)

    def _create_field_widget(self, field_name: str, field_type: Any, current_value: Any, description: str) -> Optional[Any]:
        """Create the appropriate widget for a field."""
        widget = None
        
        if field_type == bool:
            widget = Checkbox(value=bool(current_value), id=f"field_{field_name}")
        elif field_name.endswith("_KEY") or field_name.endswith("_TOKEN"):
            # API keys and tokens
            widget = Input(
                value=str(current_value) if current_value else "",
                password=True,
                placeholder=description,
                id=f"field_{field_name}",
                classes="field-input-compact"
            )
        elif field_name in ["COGNITIVE_MODEL", "VALIDATION_MODEL", "SUMMARY_MODEL", "FALLBACK_MODEL"]:
            model_choices = get_all_available_model_names()
            try:
                select_value = current_value if any(c[1] == current_value for c in model_choices) else Select.BLANK
                widget = Select(
                    options=model_choices,
                    value=select_value,
                    prompt=description,
                    id=f"field_{field_name}",
                    allow_blank=True,
                    classes="field-select-compact"
                )
            except Exception as e: 
                self.app.bell()
                widget = Input(
                    value=str(current_value) if current_value is not None else "", 
                    placeholder=description, 
                    id=f"field_{field_name}",
                    classes="field-input-compact"
                )
        else: 
            widget = Input(
                value=str(current_value) if current_value is not None else "",
                placeholder=description,
                id=f"field_{field_name}",
                classes="field-input-compact"
            )
        
        return widget

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save_env":
            if not self.env_data:
                self.query_one("#fixed_status_message", Static).update("[b red]No data loaded to save.[/b red]")
                self.app.bell()
                return

            updated_values: Dict[str, Any] = {}
            # Collect current values from widgets
            for field_name, widget in self._form_widgets.items():
                field_info = EnvConfigModel.model_fields[field_name]
                field_type = field_info.annotation
                
                if isinstance(widget, Checkbox):
                    updated_values[field_name] = widget.value
                elif isinstance(widget, Input):
                    raw_value = widget.value.strip() if widget.value else ""
                    
                    # Convert empty strings to None for optional fields
                    if not raw_value:
                        updated_values[field_name] = None
                    else:
                        # Type conversion for input fields
                        if field_type == float:
                            try:
                                updated_values[field_name] = float(raw_value)
                            except ValueError:
                                updated_values[field_name] = field_info.default
                        elif field_type == int:
                            try:
                                updated_values[field_name] = int(float(raw_value))
                            except ValueError:
                                updated_values[field_name] = field_info.default
                        else:
                            updated_values[field_name] = raw_value
                elif isinstance(widget, Select):
                    updated_values[field_name] = widget.value if widget.value != Select.BLANK else None
            
            try:
                # Create a new EnvConfigModel instance with the updated values
                # This will trigger Pydantic validation.
                new_env_config = EnvConfigModel(**updated_values)

                backup_msg = ""
                if backup_dotenv_file():
                    backup_msg = f" Backup created ({DOTENV_PATH.name}.bak)."

                # Save each field to .env
                something_changed = False
                save_errors = []
                for field_name in new_env_config.model_fields.keys():
                    new_value = getattr(new_env_config, field_name)
                    current_raw_value = self.initial_raw_values.get(field_name)
                    
                    # Prepare value for saving (convert to string representation)
                    value_to_save_str: Optional[str] = None
                    if new_value is None:
                        value_to_save_str = None
                    else:
                        value_to_save_str = str(new_value)

                    # Only save if the value has actually changed
                    if str(current_raw_value or '') != str(value_to_save_str or ''):
                        try:
                            save_dotenv_value(field_name, value_to_save_str)
                            something_changed = True
                        except Exception as e_save:
                            save_errors.append(f"Failed to save '{field_name}': {e_save}")                            
                
                if save_errors:
                    errors_str = "\n".join(save_errors)
                    # Using post_message for EnvOpComplete to update #fixed_status_message
                    self.post_message(self.EnvOpComplete(False, f"Some values failed to save:{backup_msg}\n{errors_str}"))                   
                elif something_changed:
                    self.post_message(self.EnvOpComplete(True, f"Successfully saved .env changes!{backup_msg} Restart Hephia."))
                    await self.load_env_data() 
                else:
                    self.post_message(self.EnvOpComplete(True, f"No changes detected to save.{backup_msg}"))

            except ValidationError as e:
                self.post_message(self.EnvOpComplete(False, f"Validation Error:\n{e}"))
                self.app.bell()
            except Exception as e:
                self.post_message(self.EnvOpComplete(False, f"An unexpected error occurred: {e}"))
                self.app.bell()

        elif event.button.id == "reload_env":
            await self.load_env_data() # Reload data from disk
            # Also clear any previous general status messages
            self.query_one("#fixed_status_message", Static).update("Reloaded values from .env file.")

    @on(EnvOpComplete)
    def on_env_op_complete(self, message: EnvOpComplete) -> None:
        """Display operation completion status."""
        status_widget = self.query_one("#fixed_status_message", Static)
        if message.success:
            status_widget.update(f"[b green]{message.message_text}[/b green]")
        else:
            status_widget.update(f"[b red]{message.message_text}[/b red]")
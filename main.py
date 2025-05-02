"""
Main entry point for Hephia.
Initializes and runs the complete system with all necessary checks and monitoring.
"""

from __future__ import annotations
import asyncio
import uvicorn
import threading
from dotenv import load_dotenv
import os
import sys
from datetime import datetime
from pathlib import Path

from core.server import HephiaServer
from config import Config, ProviderType
from loggers import LogManager
from event_dispatcher import global_event_dispatcher
from display.hephia_tui import handle_cognitive_event, handle_state_event, start_monitor

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
LogManager.setup_logging()


def setup_data_directory() -> None:
    """Ensure data directory exists."""
    Path('data').mkdir(exist_ok=True)


def validate_configuration() -> bool:
    """Validate LLM configuration and environment variables."""
    errors: list[str] = []

    # Map providers to their environment variable names
    provider_env_vars: dict[ProviderType, str] = {
        ProviderType.OPENPIPE: "OPENPIPE_API_KEY",
        ProviderType.OPENAI: "OPENAI_API_KEY",
        ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
        ProviderType.GOOGLE: "GOOGLE_API_KEY",
        ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
        ProviderType.PERPLEXITY: "PERPLEXITY_API_KEY",
        ProviderType.CHAPTER2: "CHAPTER2_API_KEY",
        # Note: If you have two different keys for CHAPTER2, adjust the mapping accordingly.
        ProviderType.CHAPTER2: "CHAPTER2_SOCKET_PATH"
    }

    # Validate model configurations
    for role in ['cognitive', 'validation', 'fallback', 'summary']:
        model_name = getattr(Config, f'get_{role}_model')()
        if model_name not in Config.AVAILABLE_MODELS:
            errors.append(f"Invalid {role} model configuration: {model_name}")
            continue

        model_config = Config.AVAILABLE_MODELS[model_name]
        env_var = provider_env_vars.get(model_config.provider)
        if not env_var or not os.getenv(env_var):
            errors.append(f"Missing {env_var} for {role} model ({model_name})")

    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  • {error}")
        return False

    return True


async def main() -> None:
    """Initialize and run the complete Hephia system."""
    print(f"""
╔═══════════════════════════╗
║    Hephia Project v0.2    ║
╚═══════════════════════════╝
Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

    # Load environment variables
    load_dotenv()

    Config.load_user_models()

    if not validate_configuration():
        return
    
    headless = Config.get_headless()

    # Setup directory structure
    setup_data_directory()

    print("\nInitializing systems...")

    try:
        # Initialize server using the async factory method
        server = await HephiaServer.create()

        # Start interface in a separate thread
        if not headless:
            vis_thread = threading.Thread(target=start_monitor, daemon=True)
            vis_thread.start()

            # Hook up event handlers
            global_event_dispatcher.add_listener(
                "cognitive:context_update",
                lambda event: handle_cognitive_event(event)
            )
            global_event_dispatcher.add_listener(
                "state:changed",
                lambda event: handle_state_event(event)
            )
        else:
            print("gui disabled; using manual printouts of major activity")
            global_event_dispatcher.add_listener(
                "cognitive:context_update",
                lambda event: print_cognitive_event(event)
            )

        print("""
Hephia is now active! 

Press Ctrl+C to shutdown gracefully
        """)

        # Configure and run FastAPI server via uvicorn
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=5517,
            reload=Config.DEBUG if hasattr(Config, 'DEBUG') else False,
            log_level="info"
        )
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

    except KeyboardInterrupt:
        print("\n\n🌙 Shutting down Hephia...")
        await server.shutdown()
        print("Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {e}")
        raise

async def shutdown_all_tasks():
    # Get a set of all tasks (excluding the current one)
    tasks = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    if tasks:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

def print_cognitive_event(event):
    """Print cognitive event to console in a format similar to the GUI."""
    try:
        data = event.data
        if data.get('source') == 'exo_processor':
            print("\n" + "="*80)
            print("COGNITIVE UPDATE:")
            print("="*80)
            
            # Print recent messages
            messages = data.get('raw_state', [])[-2:]
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                display_name = "EXO-PROCESSOR" if role == 'user' else Config.get_cognitive_model()
                print(f"\n{display_name}:")
                print("-" * len(display_name))
                print(content)
            
            # Print summary
            summary = data.get('processed_state', 'No summary available')
            print("\nSUMMARY:")
            print("-" * 7)
            print(summary)
            print("="*80 + "\n")
    except Exception as e:
        print(f"Error printing cognitive event: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # This ensures that any lingering tasks are cancelled.
        loop = asyncio.get_running_loop()
        loop.run_until_complete(shutdown_all_tasks())

"""
Main entry point for Hephia.
Initializes and runs the complete system with all necessary checks and monitoring.
"""

import asyncio
import uvicorn
import threading
from dotenv import load_dotenv
import os
from datetime import datetime
from pathlib import Path

from core.server import HephiaServer
from config import Config, ProviderType
from loggers import LogManager
from event_dispatcher import global_event_dispatcher
from display.hephia_tui import handle_cognitive_event, handle_state_event, start_monitor

LogManager.setup_logging()

def setup_data_directory():
    """Ensure data directory exists."""
    Path('data').mkdir(exist_ok=True)

def validate_configuration():
    """Validate LLM configuration and environment variables."""
    errors = []
    
    # Map providers to their environment variable names
    provider_env_vars = {
        ProviderType.OPENPIPE: "OPENPIPE_API_KEY",
        ProviderType.OPENAI: "OPENAI_API_KEY",
        ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
        ProviderType.GOOGLE: "GOOGLE_API_KEY",
        ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
        ProviderType.PERPLEXITY: "PERPLEXITY_API_KEY"
    }
    
    # Validate model configurations
    for role in ['cognitive', 'validation', 'fallback']:
        model_name = getattr(Config, f'get_{role}_model')()
        
        if model_name not in Config.AVAILABLE_MODELS:
            errors.append(f"Invalid {role} model configuration: {model_name}")
            continue
            
        model_config = Config.AVAILABLE_MODELS[model_name]
        env_var = provider_env_vars[model_config.provider]
        
        if not os.getenv(env_var):
            errors.append(f"Missing {env_var} for {role} model ({model_name})")
    
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    return True

async def main():
    """Initialize and run the complete Hephia system."""
    print(f"""
╔═══════════════════════════════════════════════╗
║             Hephia Project v0.2               ║
║     A Digital Homunculus in Latent Space      ║
╚═══════════════════════════════════════════════╝
Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # Load environment variables
    load_dotenv()
    
    if not validate_configuration():
        return
    
    # Setup directory structure
    setup_data_directory()
    
    print("\nInitializing systems...")
    
    try:
        # Initialize server
        server = HephiaServer()

        # Start interface in a separate thread
        vis_thread = threading.Thread(target=start_monitor, daemon=True)
        vis_thread.start()

        # Hook up event handlers
        # Cognitive context handler
        global_event_dispatcher.add_listener(
            "cognitive:context_update",
            lambda event: handle_cognitive_event(event)
        )

        # System state handler 
        global_event_dispatcher.add_listener(
            "state:changed",
            lambda event: handle_state_event(event)
        )
        
        print("""
Hephia is now active! 

Press Ctrl+C to shutdown gracefully
        """)

        # Configure and run FastAPI server
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=8000,
            reload=Config.DEBUG if hasattr(Config, 'DEBUG') else False,
            log_level="info"
        )
        
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()
        
    except KeyboardInterrupt:
        print("\n\n🌙 Shutting down Hephia...")
        await server.shutdown()
        vis_thread.stop()
        print("Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
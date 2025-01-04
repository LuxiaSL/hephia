"""
Main entry point for Hephia.
Initializes and runs the complete system with all necessary checks and monitoring.
"""

import asyncio
import uvicorn
from dotenv import load_dotenv
import os
from datetime import datetime
from pathlib import Path

from core.server import HephiaServer
from config import Config
from loggers import LogManager
LogManager.setup_logging()

def setup_data_directory():
    """Ensure data directory exists."""
    Path('data').mkdir(exist_ok=True)

async def main():
    """Initialize and run the complete Hephia system."""
    print(f"""
╔═══════════════════════════════════════════════╗
║             Hephia Project v0.1               ║
║     A Digital Homunculus in Latent Space     ║
╚═══════════════════════════════════════════════╝
Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables and their purposes
    required_vars = {
        'OPENROUTER_API_KEY': 'OpenRouter API key for command validation',
        'PERPLEXITY_API_KEY': 'Perplexity API key for search functionality',
        'OPENPIPE_API_KEY': 'OpenPipe API key for cognitive LLM',
        'OPENPIPE_MODEL': 'Model ID for cognitive LLM',
        'OPENROUTER_MODEL': 'Model ID for command validation'
    }
    
    missing = [f"{var} ({desc})" for var, desc in required_vars.items() if not os.getenv(var)]
    if missing:
        print("\nMissing required environment variables:")
        for var in missing:
            print(f"  • {var}")
        return
    
    # Setup directory structure
    setup_data_directory()
    
    print("\nInitializing systems...")
    
    try:
        # Initialize server
        server = HephiaServer()
        
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
        print("Goodbye! Thank you for witnessing the emergence.\n")
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
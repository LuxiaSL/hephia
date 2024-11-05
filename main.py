# main.py

import asyncio
from pet import Pet

async def main():
    # Initialize the pet instance
    pet = Pet()
    
    # Start the pet's main loop (async, using the global timer)
    await pet.start()

    # Keep the script running to observe output (or implement further CLI interactions)
    try:
        while pet.is_active:
            # Placeholder for monitoring; can print state periodically or wait for commands
            await asyncio.sleep(1)  # Keeps the loop alive without blocking
    except KeyboardInterrupt:
        print("Shutting down pet...")
    finally:
        pet.shutdown()  # Graceful shutdown

if __name__ == '__main__':
    asyncio.run(main())

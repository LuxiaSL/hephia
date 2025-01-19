# unix_bridge_aiohttp.py

import uvicorn
from fastapi import FastAPI, Request, HTTPException
import logging
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("unix_bridge_aiohttp")

# Path to the Unix socket
SOCKET_PATH = "/path/to/socket"
UNIX_ENDPOINT_ROOT = f"http://localhost:6006"  

app = FastAPI()

# Custom connector for Unix socket
class UnixConnector(aiohttp.connector.BaseConnector):
    def __init__(self, socket_path, *args, **kwargs):
        self.socket_path = socket_path
        super().__init__(*args, **kwargs)

    async def connect(self, req, *args, **kwargs):
        return await super().connect(req, *args, **kwargs)

@app.post("/v1/chat/completions")
async def forward_chat_completions(request: Request):
    # Extract JSON payload from the incoming request
    try:
        payload = await request.json()
        logger.debug(f"Received payload: {payload}")
    except Exception as e:
        logger.error(f"Failed to parse JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Forward the request to Unix socket using aiohttp
    try:
        connector = aiohttp.UnixConnector(path=SOCKET_PATH)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post("/v1/chat/completions", json=payload) as response:
                status = response.status
                response_json = await response.json()
                logger.debug(f"Received response status: {status}")
                logger.debug(f"Received response from server: {response_json}")
                if status == 200:
                    return response_json
                elif status == 429:
                    retry_after = response.headers.get('Retry-After', '1')
                    logger.warning(f"Rate limited by server, retry after {retry_after}s")
                    raise HTTPException(status_code=429, detail="Rate limited by server")
                else:
                    logger.error(f"Server returned error status: {status}")
                    raise HTTPException(status_code=500, detail="Error from server")
    except aiohttp.ClientError as e:
        logger.error(f"Error forwarding request to server: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with server: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

if __name__ == "__main__":
    # Start the bridge on TCP port 6005 with detailed logs
    uvicorn.run(app, host="0.0.0.0", port=6005, log_level="debug")


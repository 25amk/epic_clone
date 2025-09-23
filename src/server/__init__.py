"""
src.server: The main FastAPI server APP
"""
from loguru import logger
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from chainlit.utils import mount_chainlit
from contextlib import asynccontextmanager
from src.config import get_settings
from src.server.chain_cache import get_chat_chain

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force the chatchain to load on boot.
    chain = get_chat_chain()
    yield

app = FastAPI(lifespan = lifespan)

# FIXME: target should be auto derived from __file__
mount_chainlit(app=app, target="src/server/chatui.py", path=f"{settings.base_path}/epicui")

# TODO: Other apps
@app.get(f"{settings.base_path}/app")
def read_main():
    return {"message": "Hello World from main app"}


# Fallback
@app.get("/{path:path}")
async def redirect(path):
    return RedirectResponse(url = f'{settings.base_path}/epicui/')


from fastapi import FastAPI
from settings import config
from fastapi.responses import FileResponse
import asyncio
from ws import router, manager
from ws.upstream import receiver_loop

app = FastAPI(title="Hubble Chatbot Backend")


@app.on_event("startup")
async def startup_tasks():
    if config.ws_upstream_url:
        asyncio.create_task(receiver_loop(config.ws_upstream_url, manager))


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}


@app.get("/test/ws")
async def get_ws_test():
    return FileResponse("static/ws.html")


app.include_router(router)

from fastapi import FastAPI
from settings import config
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import HTMLResponse
import asyncio
import json
from contextlib import suppress
try:
    import websockets
except Exception:  # optional during type-check
    websockets = None

app = FastAPI(title="Hubble Chatbot Backend")


# Simple HTML to test WS rooms
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Rooms WS Test</title>
    </head>
    <body>
        <h1>WebSocket Rooms</h1>
        <form action="" onsubmit="connect(event)">
            <input type="text" id="room" placeholder="room" autocomplete="off"/>
            <button>Connect</button>
        </form>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = null;
            function connect(event) {
                var room = document.getElementById('room').value || 'lobby';
                if (ws) { ws.close(); }
                ws = new WebSocket(`ws://` + location.host + `/ws/rooms/` + encodeURIComponent(room));
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
                event.preventDefault();
            }
            function sendMessage(event) {
                if (!ws) { alert('connect first'); event.preventDefault(); return; }
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


class RoomConnectionManager:
    def __init__(self):
        self.room_to_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, room: str, websocket: WebSocket):
        await websocket.accept()
        if room not in self.room_to_connections:
            self.room_to_connections[room] = set()
        self.room_to_connections[room].add(websocket)

    def disconnect(self, room: str, websocket: WebSocket):
        room_set = self.room_to_connections.get(room)
        if not room_set:
            return
        if websocket in room_set:
            room_set.remove(websocket)
        if not room_set:
            del self.room_to_connections[room]

    async def send_personal(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, room: str, message: str):
        connections = self.room_to_connections.get(room, set())
        for ws in list(connections):
            try:
                await ws.send_text(message)
            except Exception:
                # Drop broken sockets
                self.disconnect(room, ws)


manager = RoomConnectionManager()
router = APIRouter()


async def _upstream_receiver_loop(url: str):
    if websockets is None:
        return
    backoff = 1
    while True:
        try:
            async with websockets.connect(url) as ws:
                backoff = 1
                async for msg in ws:
                    room = "lobby"
                    payload = msg
                    try:
                        data = json.loads(msg)
                        if isinstance(data, dict):
                            room = data.get("room") or room
                            payload = data.get("message") or payload
                    except Exception:
                        pass
                    await manager.broadcast(room, str(payload))
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


@app.on_event("startup")
async def startup_tasks():
    if config.ws_upstream_url:
        asyncio.create_task(_upstream_receiver_loop(config.ws_upstream_url))


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}


@app.get("/test/ws")
async def get_ws_test():
    return HTMLResponse(html)


@router.websocket("/ws/rooms/{room}")
async def websocket_room(websocket: WebSocket, room: str):
    await manager.connect(room, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # echo and broadcast to room
            await manager.send_personal(f"You wrote: {data}", websocket)
            await manager.broadcast(room, f"[{room}] {data}")
    except WebSocketDisconnect:
        manager.disconnect(room, websocket)
        await manager.broadcast(room, f"[system] a client left {room}")


@router.post("/rooms/{room}/publish")
async def publish_room_message(room: str, message: str):
    await manager.broadcast(room, message)
    return {"ok": True}


app.include_router(router)

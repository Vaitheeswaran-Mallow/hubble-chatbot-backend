from typing import Dict, Set
from fastapi import WebSocket


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
                self.disconnect(room, ws)


manager = RoomConnectionManager() 
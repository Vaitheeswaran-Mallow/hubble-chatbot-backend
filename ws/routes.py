from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .manager import manager


router = APIRouter()


@router.websocket("/ws/rooms/{room}")
async def websocket_room(websocket: WebSocket, room: str):
    await manager.connect(room, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal(f"You wrote: {data}", websocket)
            # removed broadcast of incoming client messages to avoid duplicating questions in the room
    except WebSocketDisconnect:
        manager.disconnect(room, websocket)
        await manager.broadcast(room, f"[system] a client left {room}")


@router.post("/rooms/{room}/publish")
async def publish_room_message(room: str, message: str):
    await manager.broadcast(room, message)
    return {"ok": True} 
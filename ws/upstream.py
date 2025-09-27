import asyncio
import json
try:
    import websockets
except Exception:
    websockets = None


async def receiver_loop(url: str, manager):
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
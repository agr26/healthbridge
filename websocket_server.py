from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any
import logging
import json
import asyncio
from datetime import datetime

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.logger = logging.getLogger('WebSocketManager')

    async def connect(self, websocket: WebSocket, client_id: str, data_type: str):
        await websocket.accept()
        if data_type not in self.active_connections:
            self.active_connections[data_type] = set()
        self.active_connections[data_type].add(websocket)
        self.logger.info(f"Client {client_id} connected to {data_type} stream")

    def disconnect(self, websocket: WebSocket, data_type: str):
        if data_type in self.active_connections:
            self.active_connections[data_type].discard(websocket)

    async def broadcast(self, data_type: str, message: Dict[str, Any]):
        if data_type in self.active_connections:
            for connection in self.active_connections[data_type]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    self.logger.error(f"Error broadcasting to client: {str(e)}")
                    await self.disconnect(connection, data_type)

def setup_websocket_routes(app: FastAPI, manager: WebSocketManager):
    @app.websocket("/ws/{client_id}/{data_type}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str, data_type: str):
        await manager.connect(websocket, client_id, data_type)
        try:
            while True:
                data = await websocket.receive_text()
                # Process incoming messages if needed
                await manager.broadcast(data_type, {"message": "Update received", "data": data})
        except WebSocketDisconnect:
            manager.disconnect(websocket, data_type)

    return app
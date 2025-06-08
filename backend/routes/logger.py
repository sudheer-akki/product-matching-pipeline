# backend/routes/logger.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from core import log_event

router = APIRouter()

class LogEntry(BaseModel):
    level: str
    message: str
    metadata: dict = {}

@router.post("/log")
async def log_from_frontend(entry: LogEntry, request: Request):
    request_id = request.headers.get("x-request-id", "frontend")
    log_event(entry.level, entry.message, {**entry.metadata, "source": "frontend", "request_id": request_id})
    return {"status": "ok"}

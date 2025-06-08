from fastapi import APIRouter
from core import log_event
from . import image_search, text_search, logger 

router = APIRouter()

try:
    print("→ registering image_search")
    router.include_router(image_search.router)
    log_event("info", "Image search router registered", {"route": "/search/image"})
except Exception as e:
    log_event("error", "Failed to register image_search router", {"exception": str(e)})

try:
    print("→ registering text_search")
    router.include_router(text_search.router)
    log_event("info", "Text search router registered", {"route": "search/text"})
except Exception as e:
    log_event("error", "Failed to register text_search router", {"exception": str(e)})

try:
    print("→ registering Log Router")
    router.include_router(logger.router)
    log_event("info", "Log router registered", {"route": "/log"})
except Exception as e:
    log_event("error", "Failed to register Log router", {"exception": str(e)})

import logging
from fastapi import APIRouter
from fastapi.responses import RedirectResponse


general = APIRouter()


@general.get("/heartbeat", tags=["general"], description="Return True if the API is alive and running")
async def heartbeat():
    logging.info("GET request on /heartbeat")
    return True


@general.get("/", tags=["general"], description="Redirection to /docs")
async def redirect_to_docs():
    logging.info("GET request on / -> redirecting to /docs")
    return RedirectResponse("/docs")

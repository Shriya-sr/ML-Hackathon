"""PathSnap FastAPI backend (Laptop 1).

Run:  python -m backend.main            (binds PATHSNAP_HOST:PATHSNAP_PORT, default 0.0.0.0:8000)
Docs: http://<laptop-1-ip>:8000/docs

Every request is logged to the console (client IP, method, path, status,
duration) so the evaluator can watch inference requests arrive on Laptop 1.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402

from backend.dependencies import init_state  # noqa: E402
from backend.routers import system, trips  # noqa: E402
from backend.settings import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="PathSnap API", version=system.API_VERSION, lifespan=lifespan,
                  description="Map-matching and route reconstruction. Reference routes stay on the server.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], allow_credentials=False)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        client = request.client.host if request.client else "?"
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error %s %s", request.method, request.url.path)
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path, response.status_code,
                 1000 * (time.perf_counter() - t0))
        return response

    app.include_router(system.router)
    app.include_router(trips.router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    log.info("starting PathSnap API on http://%s:%d", s.host, s.port)
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning")

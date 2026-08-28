"""FastAPI app. Run from the project root:

    uvicorn webapp.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .config import ProjectDataError
from .inference import get_bundle

HERE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both experiments now so a missing file shows up at startup.
    for key in config.EXPERIMENTS:
        try:
            get_bundle(key)
        except ProjectDataError as exc:
            print(f"[warn] {key} unavailable: {exc}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Redshift-based Astronomical Object Classification",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


@app.exception_handler(ProjectDataError)
async def handle_project_data_error(request: Request, exc: ProjectDataError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Redshift-based Astronomical Object Classification",
            "task": config.TASK,
            "experiments": [
                {"key": e.key, "label": e.label} for e in config.EXPERIMENTS.values()
            ],
            "default_experiment": config.DEFAULT_EXPERIMENT,
            "z_min": config.Z_MIN,
            "z_max": config.Z_MAX,
            "z_step": config.Z_STEP,
        },
    )


@app.get("/api/experiments")
def list_experiments():
    return {
        "experiments": [get_bundle(key).summary() for key in config.EXPERIMENTS],
        "default": config.DEFAULT_EXPERIMENT,
        "z_min": config.Z_MIN,
        "z_max": config.Z_MAX,
    }


@app.get("/api/predict")
def predict(
    z: float = Query(..., ge=config.Z_MIN, le=config.Z_MAX),
    experiment: str = Query(config.DEFAULT_EXPERIMENT),
):
    if experiment not in config.EXPERIMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment}")

    bundle = get_bundle(experiment)
    return {
        "z": z,
        "experiment": bundle.summary(),
        "predictions": [p.as_dict() for p in bundle.predict(z)],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "experiments": list(config.EXPERIMENTS)}

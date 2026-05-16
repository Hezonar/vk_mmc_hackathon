from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes_html import router as html_router
from app.api.routes_json import router as json_router
from app.api.routes_profpath import router as profpath_router
from app.domain.errors import AnalyzerExecutionError, AnalyzerNotFoundError
from app.services.prediction import PredictionService

BASE_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    app = FastAPI(title="ПрофАрбитр AI", version="0.1.0")
    templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))
    app.state.templates = templates
    app.state.prediction_service = PredictionService()

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_htmx(request: Request, exc: RequestValidationError):
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request,
                "partials/errors.html",
                {"errors": exc.errors()},
                status_code=422,
            )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(AnalyzerNotFoundError)
    async def analyzer_not_found(request: Request, exc: AnalyzerNotFoundError):
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request,
                "partials/errors.html",
                {
                    "errors": [{"type": "not_found", "loc": ("analyzer_id",), "msg": str(exc)}],
                },
                status_code=404,
            )
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AnalyzerExecutionError)
    async def analyzer_failed(request: Request, exc: AnalyzerExecutionError):
        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request,
                "partials/errors.html",
                {
                    "errors": [{"type": "runtime", "loc": ("analyzer",), "msg": str(exc)}],
                },
                status_code=500,
            )
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(html_router)
    app.include_router(json_router)
    app.include_router(profpath_router)
    return app


app = create_app()

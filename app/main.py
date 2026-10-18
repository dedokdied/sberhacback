"""FastAPI-приложение сервиса прогнозирования (маршруты BE-01..BE-06).

Маршруты:

* ``POST /predict`` — multipart sales+stock, полный расчёт 562 496 строк (CSV);
* ``GET /predict`` — фильтр последнего прогноза по article и/или warehouse (CSV);
* ``GET /predict/all?format=csv|json`` — весь последний прогноз (стриминг);
* ``GET /health`` — статус и версия модели без запуска расчёта.

Модель загружается на startup; при недоступных артефактах приложение не стартует.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import validators
from app.predictor import (
    CacheEmptyError,
    ForecastService,
    PredictInProgressError,
    UnknownIdError,
)
from src.predict import ModelArtifactError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "model")))

HTTP_STATUS_CODES = {
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
}


def _error_payload(detail: str, code: str) -> dict:
    return {"detail": detail, "code": code}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Загрузка модели из %s", MODEL_DIR)
    try:
        service = ForecastService(MODEL_DIR)
    except ModelArtifactError as exc:
        logger.error("Модель не загружена, сервис не стартует: %s", exc)
        raise
    logger.info(
        "Модель готова: version=%s pairs=%d",
        service.model_version,
        service.catalogs.pair_count,
    )
    app.state.service = service
    yield


app = FastAPI(title="Nonton Forecast API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(validators.ApiError)
def handle_api_error(request: Request, exc: validators.ApiError):
    if exc.status_code >= 500:
        logger.error("API error: %s", exc.detail)
    return JSONResponse(status_code=exc.status_code, content=exc.as_payload())


@app.exception_handler(RequestValidationError)
def handle_request_validation(request: Request, exc: RequestValidationError):
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        problems.append(f"{location}: {error.get('msg', 'ошибка')}")
    detail = "Ошибка валидации запроса: " + "; ".join(problems) if problems else "Ошибка валидации запроса"
    return JSONResponse(
        status_code=422, content=_error_payload(detail, "VALIDATION_ERROR")
    )


@app.exception_handler(StarletteHTTPException)
def handle_http_exception(request: Request, exc: StarletteHTTPException):
    code = HTTP_STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=_error_payload(detail, code))


@app.get("/health")
def health(request: Request) -> dict:
    service: ForecastService = request.app.state.service
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": service.model_version,
    }


@app.post("/predict")
def predict(
    request: Request,
    sales: UploadFile = File(...),
    stock: UploadFile = File(...),
):
    service: ForecastService = request.app.state.service
    sales_frame = validators.read_sales_upload(sales.file.read(), service.catalogs)
    stock_frame = validators.read_stock_upload(stock.file.read(), service.catalogs)
    try:
        snapshot = service.run_prediction(sales_frame, stock_frame)
    except PredictInProgressError as exc:
        raise validators.ConflictError(str(exc)) from exc
    headers = {
        "Content-Disposition": 'attachment; filename="forecast.csv"',
        "X-Model-Version": service.model_version,
        "X-Forecast-Rows": str(snapshot.rows),
        "X-Last-Sales-Date": snapshot.last_sales_date,
    }
    return StreamingResponse(
        service.iter_csv_chunks(snapshot.forecast),
        media_type="text/csv",
        headers=headers,
    )


@app.get("/predict")
def get_predict(
    request: Request,
    article: str | None = Query(default=None),
    warehouse: str | None = Query(default=None),
):
    service: ForecastService = request.app.state.service
    article, warehouse = validators.require_filter_params(article, warehouse)
    try:
        frame = service.select(article, warehouse)
    except CacheEmptyError as exc:
        raise validators.ConflictError(str(exc)) from exc
    except UnknownIdError as exc:
        raise validators.NotFoundError(
            f"Неизвестный {exc.kind} в каталоге модели: {exc.value!r}"
        ) from exc
    headers = {
        "Content-Disposition": 'attachment; filename="forecast_filtered.csv"',
        "X-Model-Version": service.model_version,
        "X-Forecast-Rows": str(len(frame)),
    }
    return StreamingResponse(
        service.iter_csv_chunks(frame), media_type="text/csv", headers=headers
    )


@app.get("/predict/all")
def get_predict_all(request: Request, format: str = Query(default="csv")):
    service: ForecastService = request.app.state.service
    if format not in {"csv", "json"}:
        raise validators.BadRequestError(
            f"format должен быть 'csv' или 'json', получено {format!r}"
        )
    try:
        snapshot = service.snapshot()
    except CacheEmptyError as exc:
        raise validators.ConflictError(str(exc)) from exc
    headers = {
        "X-Model-Version": service.model_version,
        "X-Forecast-Rows": str(snapshot.rows),
    }
    if format == "csv":
        headers["Content-Disposition"] = 'attachment; filename="forecast_all.csv"'
        return StreamingResponse(
            service.iter_csv_chunks(snapshot.forecast),
            media_type="text/csv",
            headers=headers,
        )
    return StreamingResponse(
        service.iter_json_stream(snapshot.forecast),
        media_type="application/json",
        headers=headers,
    )

"""Строгие проверки входных данных и единые ошибки HTTP API.

Все проверки CSV выполняются общим модулем :mod:`src.data_loader` (тот же код,
что и в offline ML). Здесь только перевод доменных исключений в HTTP-статусы:

* :class:`BadRequestError` -> 400 ``BAD_REQUEST`` (пустой/сломанный CSV, плохой query);
* :class:`ValidationError` -> 422 ``VALIDATION_ERROR`` (схема, типы, даты, история, каталог);
* :class:`NotFoundError` -> 404 ``NOT_FOUND`` (неизвестный ID в GET-фильтре);
* :class:`ConflictError` -> 409 ``CONFLICT`` (GET до POST, параллельный POST).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src import data_loader


class ApiError(Exception):
    """Базовая ошибка API с HTTP-статусом и кодом единого формата."""

    status_code = 500
    code = "INTERNAL_ERROR"
    default_detail = "Внутренняя ошибка сервиса"

    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail or self.default_detail)
        self.detail = detail or self.default_detail

    def as_payload(self) -> dict:
        return {"detail": self.detail, "code": self.code}


class BadRequestError(ApiError):
    status_code = 400
    code = "BAD_REQUEST"
    default_detail = "Некорректный запрос"


class ValidationError(ApiError):
    status_code = 422
    code = "VALIDATION_ERROR"
    default_detail = "Входные данные не прошли валидацию"


class NotFoundError(ApiError):
    status_code = 404
    code = "NOT_FOUND"
    default_detail = "Объект не найден"


class ConflictError(ApiError):
    status_code = 409
    code = "CONFLICT"
    default_detail = "Конфликт состояния сервиса"


def read_sales_upload(raw: bytes, catalogs) -> pd.DataFrame:
    """Строгая валидация sales из POST /predict (длины, схемы, дат, каталога)."""
    try:
        return data_loader.load_sales(raw, catalogs=catalogs, strict=True)
    except data_loader.CsvParseError as exc:
        raise BadRequestError(str(exc)) from exc
    except data_loader.DataValidationError as exc:
        raise ValidationError(str(exc)) from exc


def read_stock_upload(raw: bytes, catalogs) -> pd.DataFrame:
    """Строгая валидация stock из POST /predict (конфликты дублей -> 422)."""
    try:
        return data_loader.load_stock(raw, catalogs=catalogs, strict=True)
    except data_loader.CsvParseError as exc:
        raise BadRequestError(str(exc)) from exc
    except data_loader.DataValidationError as exc:
        raise ValidationError(str(exc)) from exc


def require_filter_params(article: Optional[str], warehouse: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """GET /predict: хотя бы один из параметров article/warehouse обязателен."""
    article = article.strip() if article else None
    warehouse = warehouse.strip() if warehouse else None
    if not article and not warehouse:
        raise BadRequestError(
            "Укажите хотя бы один query-параметр: article или warehouse"
        )
    return article or None, warehouse or None

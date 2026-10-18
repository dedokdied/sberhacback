"""Единый сервис прогноза: загрузка артефактов, вызов ML, in-memory кэш и стриминг.

Инварианты (раздел 6 backend-ТЗ):

* прогноз считается только при POST; после успеха кэш заменяется атомарно;
* при ошибке POST предыдущий успешный прогноз остаётся доступным;
* параллельные тяжёлые POST не выполняются: второй получает 409 без очереди;
* повторный одинаковый POST даёт идентичный результат (набор входов и модели фиксирован).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.predict import ModelArtifactError, ModelBundle, predict_64

logger = logging.getLogger(__name__)

CSV_CHUNK_ROWS = 20000
OUTPUT_COLUMNS = ["date", "warehouse_name", "article", "quantity"]


class CacheEmptyError(RuntimeError):
    """GET-запрос до первого успешного POST."""


class PredictInProgressError(RuntimeError):
    """Параллельный POST во время уже выполняющегося расчёта."""


class UnknownIdError(KeyError):
    """ID из GET-фильтра отсутствует в закреплённом каталоге."""

    def __init__(self, kind: str, value: str) -> None:
        super().__init__(value)
        self.kind = kind
        self.value = value


@dataclass(frozen=True)
class ForecastSnapshot:
    """Неизменяемый снимок последнего успешного прогноза."""

    forecast: pd.DataFrame
    created_at: str
    rows: int
    last_sales_date: str


class ForecastService:
    """Состояние процесса: предобученная модель + last-success cache."""

    def __init__(self, model_dir: Path | str) -> None:
        self.bundle: ModelBundle = ModelBundle.load(model_dir)
        self._cache: Optional[ForecastSnapshot] = None
        self._post_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return self.bundle.model_version

    @property
    def catalogs(self):
        return self.bundle.catalogs

    def has_forecast(self) -> bool:
        with self._cache_lock:
            return self._cache is not None

    def snapshot(self) -> ForecastSnapshot:
        with self._cache_lock:
            if self._cache is None:
                raise CacheEmptyError(
                    "Кэш пуст: сначала выполните успешный POST /predict"
                )
            return self._cache

    def run_prediction(self, sales: pd.DataFrame, stock: pd.DataFrame) -> ForecastSnapshot:
        """Выполняет predict_64 и атомарно публикует результат в кэш."""
        if not self._post_lock.acquire(blocking=False):
            raise PredictInProgressError(
                "Другой POST /predict уже выполняется, повторите позже"
            )
        try:
            started = time.perf_counter()
            cutoff = pd.Timestamp(sales["date"].max()).normalize()
            forecast = predict_64(sales, stock, self.bundle)
            snapshot = ForecastSnapshot(
                forecast=forecast,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                rows=int(len(forecast)),
                last_sales_date=str(cutoff.date()),
            )
            with self._cache_lock:
                self._cache = snapshot
            elapsed = time.perf_counter() - started
            logger.info(
                "POST /predict: rows=%d last_sales_date=%s model_version=%s elapsed=%.2fs",
                snapshot.rows,
                snapshot.last_sales_date,
                self.model_version,
                elapsed,
            )
            return snapshot
        finally:
            self._post_lock.release()

    def select(self, article: Optional[str], warehouse: Optional[str]) -> pd.DataFrame:
        """Фильтрация кэшированного прогноза; неизвестный ID -> :class:`UnknownIdError`."""
        snapshot = self.snapshot()
        if article is not None and article not in set(self.catalogs.articles):
            raise UnknownIdError("article", article)
        if warehouse is not None and warehouse not in set(self.catalogs.warehouses):
            raise UnknownIdError("warehouse", warehouse)

        frame = snapshot.forecast
        mask = pd.Series(True, index=frame.index)
        if article is not None:
            mask &= frame["article"] == article
        if warehouse is not None:
            mask &= frame["warehouse_name"] == warehouse
        return frame.loc[mask]

    @staticmethod
    def iter_csv_chunks(frame: pd.DataFrame, chunk_rows: int = CSV_CHUNK_ROWS) -> Iterator[bytes]:
        """Потоковая отдача CSV: заголовок + чанки по ``chunk_rows`` строк."""
        yield (",".join(OUTPUT_COLUMNS) + "\n").encode("utf-8")
        for start in range(0, len(frame), chunk_rows):
            chunk = frame.iloc[start : start + chunk_rows]
            yield chunk.to_csv(
                index=False, header=False, lineterminator="\n"
            ).encode("utf-8")

    @staticmethod
    def iter_json_stream(frame: pd.DataFrame, chunk_rows: int = CSV_CHUNK_ROWS) -> Iterator[bytes]:
        """Потоковый JSON-массив объектов без полной сериализации в память."""
        yield b"["
        first = True
        for start in range(0, len(frame), chunk_rows):
            chunk = frame.iloc[start : start + chunk_rows]
            payload = chunk.to_json(
                orient="records", force_ascii=False, double_precision=10
            ).encode("utf-8")
            inner = payload[1:-1]
            if not inner:
                continue
            if not first:
                yield b","
            yield inner
            first = False
        yield b"]"

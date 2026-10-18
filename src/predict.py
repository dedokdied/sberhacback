"""Единый инференс offline/API: ModelBundle и predict_64 (раздел 4 backend-ТЗ).

Контракт:

* ``ModelBundle.load(model_dir)`` загружает ``lightgbm.txt``, ``config.json``,
  ``catalogs.json`` (и опционально ``checksums.json``), сверяет schema/model
  version и порядок признаков;
* ``predict_64(sales, stock, bundle)`` возвращает DataFrame строго с колонками
  ``date, warehouse_name, article, quantity``: полный декартов грид
  «11 складов x 799 артикулов x 64 дня» = 562 496 строк, без NaN/Inf/отрицательных.

Postprocessing соответствует ТЗ: проверка конечности, ``clip >= 0``, без округления,
опциональное обнуление серий без продаж в окне (``zero_forecast_if_no_sales_in_window``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    HORIZON,
    KEY,
    build_history_features,
    build_horizon_matrix,
    build_output_frame,
)
from src.features import CATEGORY_SOURCE_COLUMNS

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSION = "1.0"
OUTPUT_COLUMNS = ["date", "warehouse_name", "article", "quantity"]


class ModelArtifactError(RuntimeError):
    """Отсутствующие или несовместимые артефакты модели."""


class ForecastInvariantError(RuntimeError):
    """Нарушение финальных инвариантов прогноза (раздел 9 ТЗ)."""


@dataclass(frozen=True)
class Catalogs:
    """Фиксированный каталог: 11 складов, 799 артикулов, словари категорий."""

    warehouses: list[str]
    articles: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "Catalogs":
        warehouses = [str(item) for item in payload.get("warehouses", [])]
        articles = [str(item) for item in payload.get("articles", [])]
        categories = {
            str(column): [str(value) for value in values]
            for column, values in payload.get("categories", {}).items()
        }
        if not warehouses or not articles:
            raise ModelArtifactError("catalogs.json: пустые списки warehouses/articles")
        return cls(warehouses=warehouses, articles=articles, categories=categories)

    def series_grid(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self.warehouses, self.articles], names=KEY
        )
        return index.to_frame(index=False)

    @property
    def pair_count(self) -> int:
        return len(self.warehouses) * len(self.articles)

    def category_codes(self, feature: str) -> tuple[str, list[str]]:
        source_column = CATEGORY_SOURCE_COLUMNS[feature]
        return source_column, self.categories.get(source_column, [])


@dataclass(frozen=True)
class ModelBundle:
    """Предобученная модель и сопутствующие артефакты, загруженные из ``model/``."""

    booster: lgb.Booster
    config: dict
    catalogs: Catalogs

    @property
    def model_version(self) -> str:
        return str(self.config.get("model_version", "unknown"))

    @property
    def schema_version(self) -> str:
        return str(self.config.get("schema_version", "unknown"))

    @property
    def history_days(self) -> int:
        return int(self.config.get("history_days", 180))

    @property
    def horizon(self) -> int:
        return int(self.config.get("horizon", HORIZON))

    @classmethod
    def load(cls, model_dir: Path | str) -> "ModelBundle":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        catalogs_path = model_dir / "catalogs.json"
        model_path = model_dir / "lightgbm.txt"
        for path in (config_path, catalogs_path, model_path):
            if not path.exists():
                raise ModelArtifactError(f"Не найден артефакт модели: {path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        schema_version = str(config.get("schema_version", ""))
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ModelArtifactError(
                f"Несовместимая schema_version модели: {schema_version!r}, "
                f"поддерживается {SUPPORTED_SCHEMA_VERSION!r}"
            )

        features = list(config.get("features", []))
        if features != FEATURE_NAMES:
            raise ModelArtifactError(
                "Список признаков в config.json не совпадает с кодовой схемой: "
                f"{features} != {FEATURE_NAMES}"
            )
        if list(config.get("categorical_features", [])) != CATEGORICAL_FEATURES:
            raise ModelArtifactError("categorical_features в config.json не совпадают с кодовой схемой")

        cls._verify_checksums(model_dir)

        catalogs = Catalogs.from_dict(json.loads(catalogs_path.read_text(encoding="utf-8")))
        booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        logger.info(
            "model loaded: version=%s schema=%s pairs=%d history_days=%d",
            config.get("model_version"),
            schema_version,
            catalogs.pair_count,
            int(config.get("history_days", 180)),
        )
        return cls(booster=booster, config=config, catalogs=catalogs)

    @staticmethod
    def _verify_checksums(model_dir: Path) -> None:
        checksums_path = model_dir / "checksums.json"
        if not checksums_path.exists():
            return
        expected = json.loads(checksums_path.read_text(encoding="utf-8"))
        for filename, digest in expected.items():
            path = model_dir / filename
            if not path.exists():
                raise ModelArtifactError(f"checksums.json ссылается на отсутствующий файл {filename}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ModelArtifactError(
                    f"Контрольная сумма {filename} не совпадает: {actual} != {digest}"
                )


def extract_series_statics(
    sales: pd.DataFrame, series: pd.DataFrame, catalogs: Catalogs
) -> pd.DataFrame:
    """Коды склада/артикула/категорий по каждой паре (порядок строк ``series``)."""
    out = pd.DataFrame(index=range(len(series)))

    warehouse_code = {value: index for index, value in enumerate(catalogs.warehouses)}
    article_code = {value: index for index, value in enumerate(catalogs.articles)}
    out["warehouse_code"] = series["warehouse_name"].map(warehouse_code).fillna(-1).astype(np.int32)
    out["article_code"] = series["article"].map(article_code).fillna(-1).astype(np.int32)

    for feature, source_column in CATEGORY_SOURCE_COLUMNS.items():
        if source_column not in sales.columns:
            out[feature] = -1
            continue
        subset = sales[[*KEY, source_column]].dropna(subset=[source_column])
        subset = subset[subset[source_column].astype(str).str.strip() != ""]
        subset = subset.drop_duplicates(subset=KEY, keep="first")
        mapping = {value: index for index, value in enumerate(catalogs.categories.get(source_column, []))}
        codes = series[KEY].merge(subset, on=KEY, how="left")[source_column].map(mapping)
        out[feature] = codes.fillna(-1).astype(np.int32)
    return out


def predict_64(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    bundle: ModelBundle,
) -> pd.DataFrame:
    """Полный прогноз на 64 дня: 11 x 799 x 64 = 562 496 строк.

    ``sales`` и ``stock`` — уже провалидированные и агрегированные датафреймы
    (см. :mod:`src.data_loader`). Функция детерминирована и не использует сеть.
    """
    if not isinstance(sales, pd.DataFrame) or not len(sales):
        raise ValueError("predict_64: sales пуст")
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
    if len(stock):
        stock = stock.copy()
        stock["date"] = pd.to_datetime(stock["date"], errors="raise").dt.normalize()
    else:
        stock = pd.DataFrame(columns=["date", "warehouse_name", "article", "stock"])

    cutoff = pd.Timestamp(sales["date"].max()).normalize()
    horizon = bundle.horizon
    forecast_dates = pd.date_range(
        cutoff + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    series = bundle.catalogs.series_grid()
    history = build_history_features(
        sales, stock, cutoff, series, history_days=bundle.history_days
    )
    statics = extract_series_statics(sales, series, bundle.catalogs)
    history = pd.concat(
        [history.reset_index(drop=True), statics.reset_index(drop=True)], axis=1
    )

    matrix = build_horizon_matrix(history, forecast_dates)
    raw = bundle.booster.predict(matrix.to_numpy(dtype=np.float32))

    transform = str(bundle.config.get("target_transform", "log1p"))
    if transform == "log1p":
        quantities = np.expm1(raw.astype(np.float64))
    elif transform == "none":
        quantities = raw.astype(np.float64)
    else:
        raise ModelArtifactError(f"Неизвестный target_transform в config.json: {transform!r}")
    quantities[~np.isfinite(quantities)] = 0.0
    quantities = np.clip(quantities, 0.0, None)

    postprocess = bundle.config.get("postprocess", {})
    if postprocess.get("zero_forecast_if_no_sales_in_window", False):
        cold_series = history["mean_180"].to_numpy() == 0
        quantities[np.repeat(cold_series, horizon)] = 0.0

    threshold = float(postprocess.get("positive_threshold", 0.0) or 0.0)
    if threshold > 0:
        quantities[quantities < threshold] = 0.0

    forecast = build_output_frame(series, forecast_dates, quantities)
    validate_forecast(forecast, bundle.catalogs, forecast_dates)
    return forecast


def validate_forecast(
    forecast: pd.DataFrame,
    catalogs: Catalogs,
    forecast_dates: pd.DatetimeIndex,
) -> None:
    """Финальная валидация прогноза (раздел 9 ТЗ): грид, даты, ключи, значения."""
    problems: list[str] = []
    horizon = len(forecast_dates)
    expected_rows = catalogs.pair_count * horizon

    if list(forecast.columns) != OUTPUT_COLUMNS:
        problems.append(f"колонки {list(forecast.columns)} != {OUTPUT_COLUMNS}")
    if len(forecast) != expected_rows:
        problems.append(f"строк {len(forecast)}, ожидалось {expected_rows}")

    dates = pd.to_datetime(forecast["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        problems.append("невалидные date в прогнозе")
    else:
        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        if len(unique_dates) != horizon:
            problems.append(f"уникальных дат {len(unique_dates)}, ожидалось {horizon}")
        else:
            expected = [
                str(day.date())
                for day in pd.date_range(unique_dates[0], periods=horizon, freq="D")
            ]
            actual = [str(day.date()) for day in unique_dates]
            if actual != expected:
                problems.append("даты прогноза не идут подряд")

    if forecast.duplicated(subset=OUTPUT_COLUMNS[:3]).any():
        problems.append("дубли ключей (date, warehouse_name, article)")

    quantities = forecast["quantity"].to_numpy(dtype=np.float64)
    if not np.isfinite(quantities).all():
        problems.append("NaN/Inf в quantity")
    if (quantities < 0).any():
        problems.append("отрицательные значения quantity")

    if problems:
        raise ForecastInvariantError("Прогноз не прошёл финальную валидацию: " + "; ".join(problems))

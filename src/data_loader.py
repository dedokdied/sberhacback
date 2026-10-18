"""Парсинг, агрегация и валидация входных CSV (единый код для offline ML и HTTP API).

Правила соответствуют backend-ТЗ (разделы 3.3, 8, 15):

* пустой или синтаксически некорректный CSV -> :class:`CsvParseError` (HTTP 400);
* схема, типы, даты, ключи, каталог, история -> :class:`DataValidationError` (HTTP 422);
* sales: дубли ``(date, warehouse_name, article)`` агрегируются суммой ``quantity``;
* stock: идентичные дубли удаляются, конфликтующие значения -> 422 (политика не согласована);
* offline-режим обучения использует послабления (``strict=False``): отрицательные
  остатки клиппятся, конфликтующие дубли разрешаются явно заданной агрегацией.

Текст CSV читается в строковом виде: ID сохраняются как есть, ведущие нули не теряются.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 180

KEY_COLUMNS = ["warehouse_name", "article"]
SALES_KEY = ["date", "warehouse_name", "article"]
STOCK_KEY = ["date", "warehouse_name", "article"]

SALES_REQUIRED_COLUMNS = ["date", "warehouse_name", "article", "quantity"]
SALES_NUMERIC_SUM = ["quantity", "sales_sum", "discount_sum", "volume_weight"]
SALES_NUMERIC_MEAN = ["cost_price"]
SALES_TEXT_FIRST = [
    "composite_division",
    "cities_name",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "characteristic",
]

STOCK_REQUIRED_COLUMNS = ["date", "warehouse_name", "article", "stock"]
STOCK_ARTICLE_COLUMNS = ("article", "char_article")
DATE_FORMAT = "%Y-%m-%d"

Source = Union[bytes, str, Path]


class CsvParseError(ValueError):
    """Пустой или синтаксически некорректный CSV -> HTTP 400."""


class DataValidationError(ValueError):
    """Нарушение схемы, типов, дат, ключей, каталога или глубины истории -> HTTP 422."""


def read_csv_table(raw: bytes, *, name: str) -> pd.DataFrame:
    """Читает CSV из байтов, не теряя текстовые ID, и отсеивает синтаксические проблемы."""
    if raw is None or not raw.strip():
        raise CsvParseError(f"{name}: файл пуст")
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError as exc:
        raise CsvParseError(f"{name}: файл не является UTF-8 CSV: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise CsvParseError(f"{name}: файл пуст") from exc
    except pd.errors.ParserError as exc:
        raise CsvParseError(f"{name}: синтаксически некорректный CSV: {exc}") from exc
    if df.shape[0] == 0:
        raise CsvParseError(f"{name}: файл не содержит строк данных")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _read_source(source: Source, *, name: str) -> pd.DataFrame:
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        raise TypeError(f"{name}: неподдерживаемый источник данных: {type(source)!r}")
    return read_csv_table(raw, name=name)


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: отсутствуют обязательные колонки: {missing}")


def _parse_dates(df: pd.DataFrame, name: str) -> pd.Series:
    parsed = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        examples = df.loc[bad, "date"].head(5).tolist()
        raise DataValidationError(
            f"{name}: невалидные даты (ожидается YYYY-MM-DD): {int(bad.sum())} строк, "
            f"примеры: {examples}"
        )
    return parsed


def _parse_numeric(
    df: pd.DataFrame,
    column: str,
    name: str,
    *,
    allow_negative: bool = False,
    allow_missing: bool = False,
) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    raw = df[column].astype("string")
    empty = df[column].isna() | (raw.str.strip() == "")
    unparsable = values.isna() & ~empty
    if unparsable.any():
        examples = df.loc[unparsable, column].head(5).tolist()
        raise DataValidationError(
            f"{name}: нечисловые значения в {column}: {int(unparsable.sum())} строк, "
            f"примеры: {examples}"
        )
    missing = values.isna()
    if missing.any() and not allow_missing:
        raise DataValidationError(
            f"{name}: пустые значения в {column}: {int(missing.sum())} строк"
        )
    nonfinite = ~missing & ~np.isfinite(values)
    if nonfinite.any():
        raise DataValidationError(
            f"{name}: бесконечные значения в {column}: {int(nonfinite.sum())} строк"
        )
    negative = values < 0
    if negative.any() and not allow_negative:
        raise DataValidationError(
            f"{name}: отрицательные значения в {column}: {int(negative.sum())} строк"
        )
    return values


def _check_key_columns(df: pd.DataFrame, name: str) -> None:
    for column in KEY_COLUMNS:
        values = df[column].astype("string")
        empty = values.isna() | (values.str.strip() == "")
        if empty.any():
            raise DataValidationError(
                f"{name}: пустые ключевые ID в колонке {column}: {int(empty.sum())} строк"
            )


def _check_catalog(df: pd.DataFrame, catalogs, name: str) -> None:
    if catalogs is None:
        return
    allowed_by_column = {
        "warehouse_name": getattr(catalogs, "warehouses", None),
        "article": getattr(catalogs, "articles", None),
    }
    for column, allowed in allowed_by_column.items():
        if not allowed:
            continue
        known = set(allowed)
        unknown = pd.unique(df.loc[~df[column].isin(known), column])
        if len(unknown):
            examples = [str(value) for value in unknown[:5]]
            raise DataValidationError(
                f"{name}: ID вне согласованного каталога в колонке {column}: "
                f"{len(unknown)} шт., примеры: {examples}"
            )


def _check_history(dates: pd.Series, name: str, min_history_days: int) -> None:
    span_days = int((dates.max() - dates.min()).days) + 1
    if span_days < min_history_days:
        raise DataValidationError(
            f"{name}: недостаточно истории: {span_days} календарных дней, "
            f"минимум {min_history_days}"
        )
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = expected.difference(pd.DatetimeIndex(dates.unique()))
    if len(missing):
        examples = [str(day.date()) for day in missing[:5]]
        logger.warning(
            "%s: разрывы календарного покрытия: %d из %d дней отсутствуют (примеры: %s)",
            name,
            len(missing),
            len(expected),
            examples,
        )


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Схлопывает дубли ``(date, warehouse_name, article)`` по департаментам.

    ``quantity`` и денежные колонки суммируются, ``cost_price`` усредняется,
    текстовые атрибуты берутся первыми в устойчивом порядке.
    """
    numeric_sum = [col for col in SALES_NUMERIC_SUM if col in df.columns]
    numeric_mean = [col for col in SALES_NUMERIC_MEAN if col in df.columns]
    text_first = [col for col in SALES_TEXT_FIRST if col in df.columns]
    columns = SALES_KEY + numeric_sum + numeric_mean + text_first
    out = df[columns].sort_values(SALES_KEY, kind="stable").reset_index(drop=True)

    duplicated_rows = int(out.duplicated(subset=SALES_KEY).sum())
    if duplicated_rows:
        logger.info(
            "sales: агрегация дублей по ключу %s: %d строк (quantity=sum)",
            SALES_KEY,
            duplicated_rows,
        )
        agg_map = {col: "sum" for col in numeric_sum}
        agg_map.update({col: "mean" for col in numeric_mean})
        agg_map.update({col: "first" for col in text_first})
        out = out.groupby(SALES_KEY, as_index=False, sort=True).agg(agg_map)
    return out.sort_values(SALES_KEY, kind="stable").reset_index(drop=True)


def load_sales(
    source: Source,
    catalogs=None,
    *,
    min_history_days: int = MIN_HISTORY_DAYS,
    strict: bool = True,
) -> pd.DataFrame:
    """Читает, валидирует и агрегирует sales-таблицу.

    Возвращает датафрейм с колонками ``date, warehouse_name, article, quantity``
    и (при наличии в файле) атрибутами категорий/города, отсортированный по ключу.
    ``strict=False`` (обучение): пустые значения в необязательных денежных колонках
    заполняются нулями.
    """
    df = _read_source(source, name="sales")
    _require_columns(df, SALES_REQUIRED_COLUMNS, "sales")

    df = df.copy()
    df["date"] = _parse_dates(df, "sales")
    df["quantity"] = _parse_numeric(df, "quantity", "sales")
    for column in SALES_NUMERIC_SUM[1:]:
        if column in df.columns:
            df[column] = _parse_numeric(
                df, column, "sales", allow_negative=True, allow_missing=True
            ).fillna(0.0)
    for column in SALES_NUMERIC_MEAN:
        if column in df.columns:
            df[column] = _parse_numeric(
                df, column, "sales", allow_negative=True, allow_missing=True
            )

    _check_key_columns(df, "sales")
    _check_catalog(df, catalogs, "sales")
    _check_history(df["date"], "sales", min_history_days)
    return aggregate_sales(df)


def load_stock(
    source: Source,
    catalogs=None,
    *,
    min_history_days: int = MIN_HISTORY_DAYS,
    strict: bool = True,
    resolve_conflicts: Optional[str] = None,
) -> pd.DataFrame:
    """Читает и валидирует stock-таблицу.

    ``strict=True`` (API): отрицательные остатки и конфликтующие дубли -> 422.
    ``strict=False`` (обучение): отрицательные клиппятся, конфликты разрешаются
    агрегацией ``resolve_conflicts`` (``mean``/``max``/``min``), если она задана.
    """
    df = _read_source(source, name="stock")
    if "article" not in df.columns and "char_article" in df.columns:
        df = df.rename(columns={"char_article": "article"})
    _require_columns(df, STOCK_REQUIRED_COLUMNS, "stock")

    df = df.copy()
    df["date"] = _parse_dates(df, "stock")
    df["stock"] = _parse_numeric(
        df, "stock", "stock", allow_negative=not strict, allow_missing=not strict
    )
    if not strict:
        missing = df["stock"].isna()
        if missing.any():
            logger.warning("stock: пустые остатки заполняются 0: %d строк", int(missing.sum()))
            df["stock"] = df["stock"].fillna(0.0)
        negative = df["stock"] < 0
        if negative.any():
            logger.warning("stock: отрицательные остатки клиппятся в 0: %d строк", int(negative.sum()))
            df.loc[negative, "stock"] = 0.0

    _check_key_columns(df, "stock")
    _check_catalog(df, catalogs, "stock")
    _check_history(df["date"], "stock", min_history_days)

    out = df[STOCK_KEY + ["stock"]]
    dup_mask = out.duplicated(subset=STOCK_KEY, keep=False)
    if dup_mask.any():
        duplicates = out.loc[dup_mask]
        nunique = duplicates.groupby(STOCK_KEY, sort=False)["stock"].nunique()
        conflicts = nunique[nunique > 1]
        if len(conflicts):
            if resolve_conflicts is None:
                examples = [tuple(index) for index in conflicts.index[:5]]
                raise DataValidationError(
                    f"stock: конфликтующие дубликаты на (date, warehouse_name, article): "
                    f"{len(conflicts)} групп; политика разрешения не согласована. "
                    f"Примеры: {examples}"
                )
            if resolve_conflicts not in {"mean", "max", "min"}:
                raise ValueError(f"stock: неизвестная стратегия resolve_conflicts={resolve_conflicts!r}")
            logger.warning(
                "stock: конфликтующие дубликаты (%d групп) разрешены через %s",
                len(conflicts),
                resolve_conflicts,
            )
            out = out.groupby(STOCK_KEY, as_index=False, sort=True).agg(
                stock=("stock", resolve_conflicts)
            )
        else:
            logger.info("stock: удалены идентичные дубликаты: %d строк", int(dup_mask.sum()))
            out = out.drop_duplicates(subset=STOCK_KEY, keep="first")
    return out.sort_values(STOCK_KEY, kind="stable").reset_index(drop=True)

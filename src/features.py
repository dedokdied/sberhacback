"""Единая генерация признаков для offline ML и онлайн-инференса (гарантия offline/API parity).

Схема (раздел 9 backend-ТЗ, Horizon-Aware Global Direct LightGBM):

* по каждой паре ``warehouse_name x article`` строятся исторические признаки
  строго on/до ``cutoff`` в окне последних ``history_days`` календарных дней;
* признаки будущего дня включают только номер горизонта ``h`` и календарь;
* предыдущие прогнозы как лаги не используются.

Все вычисления детерминированы: фиксированные окна, отсутствие случайности,
порядок ключей задаётся вызывающим кодом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["warehouse_name", "article"]

HISTORY_DAYS = 180
HORIZON = 64
EPS = 0.1

CALENDAR_FEATURES = ["h", "dow", "is_weekend", "day_of_month", "month", "dow_factor"]

HISTORY_SALES_FEATURES = [
    "mean_180",
    "std_180",
    "max_180",
    "mean_90",
    "mean_28",
    "mean_7",
    "sale_rate_180",
    "sale_rate_28",
    "sale_rate_7",
    "last_value",
    "days_since_last_sale",
    "trend_28_90",
]

HISTORY_STOCK_FEATURES = [
    "stock_last",
    "stock_mean_28",
    "stock_zero_rate_90",
    "days_of_cover",
]

DOW_FACTOR_COLUMNS = [f"dow_factor_{dow}" for dow in range(7)]

HISTORY_FEATURE_COLUMNS = (
    HISTORY_SALES_FEATURES + HISTORY_STOCK_FEATURES + DOW_FACTOR_COLUMNS
)

CATEGORICAL_FEATURES = [
    "warehouse_code",
    "article_code",
    "cat1_code",
    "cat2_code",
    "cat3_code",
]

FEATURE_NAMES = [
    "h",
    "dow",
    "is_weekend",
    "day_of_month",
    "month",
    "dow_factor",
    "mean_180",
    "mean_90",
    "mean_28",
    "mean_7",
    "std_180",
    "max_180",
    "sale_rate_180",
    "sale_rate_28",
    "sale_rate_7",
    "last_value",
    "days_since_last_sale",
    "trend_28_90",
    "stock_last",
    "stock_mean_28",
    "stock_zero_rate_90",
    "days_of_cover",
    "warehouse_code",
    "article_code",
    "cat1_code",
    "cat2_code",
    "cat3_code",
]

NUMERIC_FEATURE_NAMES = [name for name in FEATURE_NAMES if name not in CATEGORICAL_FEATURES]

CATEGORY_SOURCE_COLUMNS = {
    "cat1_code": "category_lvl_1",
    "cat2_code": "category_lvl_2",
    "cat3_code": "category_lvl_3",
}


def _window_summary(sales_window: pd.DataFrame, divisor: int) -> pd.DataFrame:
    """Суммарные статистики продаж за окно с фиксированным делителем календарных дней."""
    grouped = sales_window.groupby(KEY, sort=False)["quantity"]
    summary = pd.DataFrame(
        {
            "total": grouped.sum(),
            "n_sale": grouped.size(),
            "max": grouped.max(),
        }
    )
    squared = sales_window.assign(_q2=sales_window["quantity"] ** 2)
    summary["sumsq"] = squared.groupby(KEY, sort=False)["_q2"].sum()
    summary["mean"] = summary["total"] / divisor
    variance = summary["sumsq"] / divisor - summary["mean"] ** 2
    summary["std"] = np.sqrt(variance.clip(lower=0.0))
    summary["sale_rate"] = summary["n_sale"] / divisor
    return summary.drop(columns=["total", "sumsq"])


def build_history_features(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    cutoff: pd.Timestamp,
    series: pd.DataFrame,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Признаки истории по каждой паре на ``cutoff`` (в порядке строк ``series``)."""
    cutoff = pd.Timestamp(cutoff).normalize()
    window_start = cutoff - pd.Timedelta(days=history_days - 1)
    window_dates = pd.date_range(window_start, cutoff, freq="D")

    out = series.reset_index(drop=True).copy()
    sales_window = sales[(sales["date"] >= window_start) & (sales["date"] <= cutoff)]

    summary_180 = _window_summary(sales_window, history_days)
    summary_90 = _window_summary(
        sales_window[sales_window["date"] >= cutoff - pd.Timedelta(days=89)], 90
    )
    summary_28 = _window_summary(
        sales_window[sales_window["date"] >= cutoff - pd.Timedelta(days=27)], 28
    )
    summary_7 = _window_summary(
        sales_window[sales_window["date"] >= cutoff - pd.Timedelta(days=6)], 7
    )

    features = pd.DataFrame(index=summary_180.index)
    features["mean_180"] = summary_180["mean"]
    features["std_180"] = summary_180["std"]
    features["max_180"] = summary_180["max"]
    features["mean_90"] = summary_90["mean"]
    features["mean_28"] = summary_28["mean"]
    features["mean_7"] = summary_7["mean"]
    features["sale_rate_180"] = summary_180["sale_rate"]
    features["sale_rate_28"] = summary_28["sale_rate"]
    features["sale_rate_7"] = summary_7["sale_rate"]
    features["trend_28_90"] = (features["mean_28"] + EPS) / (features["mean_90"] + EPS)

    if len(sales_window):
        ordered = sales_window.sort_values(KEY + ["date"], kind="stable")
        last_sale = ordered.groupby(KEY, sort=False).tail(1).set_index(KEY)
        last_date = last_sale["date"]
        features["last_value"] = last_sale["quantity"]
        features["days_since_last_sale"] = (cutoff - last_date).dt.days
    else:
        features["last_value"] = np.nan
        features["days_since_last_sale"] = np.nan

    dow_values = window_dates.dayofweek.value_counts().sort_index()
    if len(sales_window):
        dow_raw = sales_window.assign(_dow=sales_window["date"].dt.dayofweek)
        dow_sum = dow_raw.groupby(KEY + ["_dow"], sort=False)["quantity"].sum()
        dow_mean = dow_sum.unstack("_dow")
        for dow in range(7):
            count = float(dow_values.get(dow, 0))
            if dow in dow_mean.columns and count > 0:
                features[f"dow_factor_{dow}"] = dow_mean[dow] / count
            else:
                features[f"dow_factor_{dow}"] = np.nan
    else:
        for dow in range(7):
            features[f"dow_factor_{dow}"] = np.nan

    stock_known = stock[stock["date"] <= cutoff] if len(stock) else stock
    if len(stock_known):
        ordered_stock = stock_known.sort_values(KEY + ["date"], kind="stable")
        stock_last = ordered_stock.groupby(KEY, sort=False)["stock"].last()
        stock_28 = stock_known[
            stock_known["date"] >= cutoff - pd.Timedelta(days=27)
        ].groupby(KEY, sort=False)["stock"].mean()
        stock_90 = stock_known[stock_known["date"] >= cutoff - pd.Timedelta(days=89)]
        stock_zero_90 = stock_90.assign(_zero=(stock_90["stock"] == 0).astype(float))
        stock_zero_rate = stock_zero_90.groupby(KEY, sort=False)["_zero"].mean()
        features["stock_last"] = stock_last
        features["stock_mean_28"] = stock_28
        features["stock_zero_rate_90"] = stock_zero_rate
        features["days_of_cover"] = stock_last / np.maximum(features["mean_28"].reindex(stock_last.index), EPS)
    else:
        features["stock_last"] = np.nan
        features["stock_mean_28"] = np.nan
        features["stock_zero_rate_90"] = np.nan
        features["days_of_cover"] = np.nan

    merged = out.merge(features.reset_index(), on=KEY, how="left", sort=False)

    for column in HISTORY_SALES_FEATURES + HISTORY_STOCK_FEATURES:
        if column in ("stock_last", "stock_mean_28", "stock_zero_rate_90", "days_of_cover"):
            continue
        merged[column] = merged[column].fillna(0.0)

    merged["days_since_last_sale"] = merged["days_since_last_sale"].clip(upper=history_days)
    merged["days_since_last_sale"] = merged["days_since_last_sale"].fillna(float(history_days))
    merged["trend_28_90"] = merged["trend_28_90"].fillna(1.0)

    for dow in range(7):
        column = f"dow_factor_{dow}"
        merged[column] = merged[column].fillna(0.0)
        merged[column] = merged[column] / (merged["mean_180"] + EPS)

    return merged


def build_horizon_matrix(
    history: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Матрица признаков для всех пар x всех дат горизонта (порядок: дата -> серия).

    Категориальные признаки кодируются числами (``-1`` = неизвестно); LightGBM
    получает их индексы через ``categorical_feature`` при обучении/инференсе.
    """
    n_series = len(history)
    total = n_series * len(forecast_dates)

    numeric_arrays = {
        name: np.empty(total, dtype=np.float32) for name in NUMERIC_FEATURE_NAMES
    }
    categorical_arrays = {
        name: np.empty(total, dtype=np.float32) for name in CATEGORICAL_FEATURES
    }
    static_numeric = {
        name: history[name].to_numpy(dtype=np.float32, copy=False)
        for name in NUMERIC_FEATURE_NAMES
        if name not in CALENDAR_FEATURES
    }
    static_codes = {
        name: history[name].to_numpy(dtype=np.float32, copy=False)
        for name in CATEGORICAL_FEATURES
    }

    position = 0
    for horizon, day in enumerate(forecast_dates, start=1):
        block = slice(position, position + n_series)
        numeric_arrays["h"][block] = float(horizon)
        numeric_arrays["dow"][block] = float(day.dayofweek)
        numeric_arrays["is_weekend"][block] = float(day.dayofweek >= 5)
        numeric_arrays["day_of_month"][block] = float(day.day)
        numeric_arrays["month"][block] = float(day.month)
        numeric_arrays["dow_factor"][block] = history[
            f"dow_factor_{day.dayofweek}"
        ].to_numpy(dtype=np.float32, copy=False)
        for name, values in static_numeric.items():
            numeric_arrays[name][block] = values
        for name, values in static_codes.items():
            categorical_arrays[name][block] = values
        position += n_series

    frame = {**numeric_arrays, **categorical_arrays}
    return pd.DataFrame(frame)[FEATURE_NAMES]


def build_output_frame(
    series: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    quantities: np.ndarray,
) -> pd.DataFrame:
    """Собирает submission-фрейм (date, warehouse_name, article, quantity) в порядке дата -> серия."""
    n_series = len(series)
    date_values = np.repeat(
        np.array([day.strftime("%Y-%m-%d") for day in forecast_dates]), n_series
    )
    warehouse_values = np.tile(series["warehouse_name"].to_numpy(), len(forecast_dates))
    article_values = np.tile(series["article"].to_numpy(), len(forecast_dates))
    return pd.DataFrame(
        {
            "date": date_values,
            "warehouse_name": warehouse_values,
            "article": article_values,
            "quantity": quantities.astype(np.float64),
        }
    )

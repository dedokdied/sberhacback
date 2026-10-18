"""Обучение Horizon-Aware Global Direct LightGBM и сборка артефактов ``model/``.

Скрипт offline-зоны ML (в Docker-образ не входит). Читает train-CSV, строит
обучающую выборку по календарным cutoff-ам, обучает единую детерминированную
модель с признаком горизонта ``h=1..64`` и сохраняет:

* ``model/lightgbm.txt`` — веса модели;
* ``model/config.json`` — версия, схема, список признаков, гиперпараметры;
* ``model/catalogs.json`` — фиксированные каталоги складов/артикулов/категорий;
* ``model/checksums.json`` — SHA-256 артефактов.

Запуск (из корня forecast-container):

    python -m src.train \
        --sales ../csv/sales_dataset_hakaton_train.csv \
        --stock ../csv/stock_dataset_hakaton_train.csv \
        --out model
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import data_loader
from src.features import (
    CATEGORICAL_FEATURES,
    CATEGORY_SOURCE_COLUMNS,
    FEATURE_NAMES,
    HORIZON,
    KEY,
    build_history_features,
    build_horizon_matrix,
)
from src.predict import (
    Catalogs,
    ModelBundle,
    extract_series_statics,
    predict_64,
)

logger = logging.getLogger("train")


def build_targets(
    sales: pd.DataFrame, series: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> np.ndarray:
    """Факт по всем парам x всем h горизонта; порядок h-major, как в матрице признаков."""
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = sales[(sales["date"] > cutoff) & (sales["date"] <= dates[-1])].copy()
    future["h"] = (future["date"] - cutoff).dt.days
    pivot = future.pivot_table(index=KEY, columns="h", values="quantity", aggfunc="sum")
    pivot = pivot.reindex(pd.MultiIndex.from_frame(series[KEY]), fill_value=0.0)
    pivot = pivot.reindex(columns=range(1, horizon + 1), fill_value=0.0)
    return pivot.fillna(0.0).to_numpy(dtype=np.float64).T


def build_cutoff_frame(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    series: pd.DataFrame,
    catalogs: Catalogs,
    cutoff: pd.Timestamp,
    history_days: int,
    horizon: int,
    target_transform: str = "log1p",
) -> pd.DataFrame:
    """Матрица признаков + таргет (``log1p(quantity)`` или сырой ``quantity``) для cutoff."""
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    history = build_history_features(sales, stock, cutoff, series, history_days)
    statics = extract_series_statics(sales, series, catalogs)
    history = pd.concat(
        [history.reset_index(drop=True), statics.reset_index(drop=True)], axis=1
    )
    matrix = build_horizon_matrix(history, dates)
    targets = build_targets(sales, series, cutoff, horizon).ravel()
    if target_transform == "log1p":
        targets = np.log1p(targets)
    matrix["_target"] = targets.astype(np.float32)
    return matrix


THRESHOLD_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def forecast_metrics(
    actual: np.ndarray, predicted: np.ndarray, series_ids: np.ndarray
) -> dict:
    """Пять официальных метрик кейса на общем массиве факта и прогноза."""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denominator > 0
    frame = pd.DataFrame(
        {
            "sid": series_ids[mask],
            "error": np.abs(actual[mask] - predicted[mask]) / denominator[mask] * 100.0,
        }
    )
    per_series = frame.groupby("sid")["error"].mean()
    all_ids = pd.Index(np.unique(series_ids), name="sid")
    per_series = per_series.reindex(all_ids, fill_value=0.0)

    total_actual = float(np.abs(actual).sum())
    wape = float(np.abs(actual - predicted).sum() / total_actual * 100.0) if total_actual else 0.0
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    bias = float((predicted.sum() - actual.sum()) / total_actual * 100.0) if total_actual else 0.0
    rmsle = float(
        np.sqrt(
            np.mean(
                (np.log1p(np.clip(predicted, 0.0, None)) - np.log1p(actual)) ** 2
            )
        )
    )
    return {
        "smape": round(float(per_series.mean()), 4),
        "wape": round(wape, 4),
        "rmse": round(rmse, 4),
        "bias_pct": round(bias, 4),
        "rmsle": round(rmsle, 4),
    }


def evaluate_bundle(
    sales: pd.DataFrame,
    stock: pd.DataFrame,
    bundle: ModelBundle,
    cutoff: pd.Timestamp,
) -> dict:
    """End-to-end проверка: predict_64 на cutoff, подбор порога, метрики против факта."""
    sales_train = sales[sales["date"] <= cutoff]
    stock_train = stock[stock["date"] <= cutoff]
    forecast = predict_64(sales_train, stock_train, bundle)

    series = bundle.catalogs.series_grid()
    actual = build_targets(sales, series, cutoff, bundle.horizon).ravel()
    predicted = forecast["quantity"].to_numpy(dtype=np.float64)
    series_ids = np.tile(np.arange(len(series)), bundle.horizon)

    rows = []
    for threshold in THRESHOLD_GRID:
        candidate = np.where(predicted >= threshold, predicted, 0.0)
        rows.append({"threshold": threshold, **forecast_metrics(actual, candidate, series_ids)})

    baseline_wape = rows[0]["wape"]
    eligible = [row for row in rows if row["wape"] <= baseline_wape * 1.02 + 1e-9]
    best = min(eligible, key=lambda row: row["smape"])
    return {
        "rows": int(len(forecast)),
        "chosen_threshold": best["threshold"],
        "chosen": best,
        "grid": rows,
    }


def save_checksums(model_dir: Path) -> None:
    checksums = {}
    for filename in ("lightgbm.txt", "config.json", "catalogs.json"):
        digest = hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        checksums[filename] = digest
    (model_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение LightGBM для хакатон-сервиса")
    parser.add_argument("--sales", required=True, help="CSV продаж (train)")
    parser.add_argument("--stock", required=True, help="CSV остатков (train)")
    parser.add_argument("--out", default="model", help="Директория артефактов модели")
    parser.add_argument("--history-days", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--cutoff-step-days", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=0, help="0 = все ядра")
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument(
        "--objective",
        choices=["l1", "tweedie"],
        default="l1",
        help="l1: regression_l1 на log1p(quantity); tweedie: Tweedie на сырой quantity",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    started = time.perf_counter()

    logger.info("Читаю sales: %s", args.sales)
    sales = data_loader.load_sales(args.sales, strict=False)
    logger.info("Читаю stock: %s", args.stock)
    stock = data_loader.load_stock(
        args.stock, strict=False, resolve_conflicts="mean"
    )

    warehouses = sorted(sales["warehouse_name"].unique().tolist())
    articles = sorted(sales["article"].unique().tolist())
    categories = {}
    for source_column in CATEGORY_SOURCE_COLUMNS.values():
        if source_column in sales.columns:
            values = sales[source_column].dropna().astype(str)
            values = values[values.str.strip() != ""]
            categories[source_column] = sorted(values.unique().tolist())
    catalogs = Catalogs(warehouses=warehouses, articles=articles, categories=categories)
    logger.info(
        "Каталог: %d складов x %d артикулов = %d пар (в train наблюдалось %d)",
        len(warehouses),
        len(articles),
        catalogs.pair_count,
        sales.groupby(KEY, sort=False).ngroups,
    )

    series = (
        sales[KEY].drop_duplicates().sort_values(KEY, kind="stable").reset_index(drop=True)
    )

    min_date = sales["date"].min().normalize()
    max_date = sales["date"].max().normalize()
    first_cutoff = min_date + pd.Timedelta(days=args.history_days - 1)
    last_cutoff = max_date - pd.Timedelta(days=args.horizon)
    cutoffs = list(
        pd.date_range(first_cutoff, last_cutoff, freq=f"{args.cutoff_step_days}D")
    )
    if len(cutoffs) < 2:
        logger.error(
            "Недостаточно истории для обучения: %s..%s, cutoff-ов %d",
            min_date.date(),
            max_date.date(),
            len(cutoffs),
        )
        return 2
    valid_cutoff = cutoffs[-1]
    train_cutoffs = cutoffs[:-1]
    logger.info(
        "Cutoff-ы: train %d (с %s по %s), valid %s",
        len(train_cutoffs),
        train_cutoffs[0].date(),
        train_cutoffs[-1].date(),
        valid_cutoff.date(),
    )

    target_transform = "log1p" if args.objective == "l1" else "none"
    frames = []
    for index, cutoff in enumerate(train_cutoffs, start=1):
        frame = build_cutoff_frame(
            sales,
            stock,
            series,
            catalogs,
            cutoff,
            args.history_days,
            args.horizon,
            target_transform=target_transform,
        )
        frames.append(frame)
        logger.info(
            "  cutoff %d/%d %s: строк %d",
            index,
            len(train_cutoffs),
            cutoff.date(),
            len(frame),
        )
    valid_frame = build_cutoff_frame(
        sales,
        stock,
        series,
        catalogs,
        valid_cutoff,
        args.history_days,
        args.horizon,
        target_transform=target_transform,
    )

    train_frame = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    train_rows = int(len(train_frame))
    logger.info("Обучающая выборка: %d строк x %d признаков", train_rows, len(FEATURE_NAMES))

    params = {
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": args.num_threads,
        "verbosity": -1,
    }
    if args.objective == "l1":
        params["objective"] = "regression_l1"
        params["metric"] = "mae"
    else:
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = 1.3
        params["metric"] = "l2"

    train_matrix = train_frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    train_label = train_frame["_target"].to_numpy(dtype=np.float32)
    del train_frame
    gc.collect()
    valid_matrix = valid_frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    valid_label = valid_frame["_target"].to_numpy(dtype=np.float32)
    del valid_frame
    gc.collect()

    train_dataset = lgb.Dataset(
        train_matrix,
        label=train_label,
        feature_name=FEATURE_NAMES,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=True,
        params=params,
    )
    train_dataset.construct()
    del train_matrix, train_label
    gc.collect()
    valid_dataset = lgb.Dataset(
        valid_matrix,
        label=valid_label,
        feature_name=FEATURE_NAMES,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_dataset,
        free_raw_data=True,
        params=params,
    )

    logger.info("Старт обучения: rounds<=%d, early_stopping=%d", args.num_rounds, args.early_stopping)
    booster = lgb.train(
        params,
        train_dataset,
        num_boost_round=args.num_rounds,
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )
    best_iteration = int(booster.best_iteration or booster.current_iteration())
    logger.info("Обучение завершено: best_iteration=%d", best_iteration)

    model_dir = Path(args.out)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "lightgbm.txt").write_text(booster.model_to_string(), encoding="utf-8")

    catalogs_payload = {
        "warehouses": catalogs.warehouses,
        "articles": catalogs.articles,
        "categories": catalogs.categories,
        "grid_pairs": catalogs.pair_count,
        "source": {
            "sales_rows_aggregated": int(len(sales)),
            "date_min": str(min_date.date()),
            "date_max": str(max_date.date()),
            "observed_pairs": int(len(series)),
        },
    }
    (model_dir / "catalogs.json").write_text(
        json.dumps(catalogs_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = {
        "model_version": args.model_version,
        "schema_version": "1.0",
        "model_type": "lightgbm-horizon-aware-global-direct",
        "objective_kind": args.objective,
        "target_transform": target_transform,
        "history_days": args.history_days,
        "horizon": args.horizon,
        "min_history_days": data_loader.MIN_HISTORY_DAYS,
        "features": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "postprocess": {
            "clip_min": 0,
            "zero_forecast_if_no_sales_in_window": True,
            "positive_threshold": 0.0,
        },
        "lgb_params": params,
        "num_boost_round": args.num_rounds,
        "best_iteration": best_iteration,
        "train_rows": train_rows,
        "train_pairs": int(len(series)),
        "grid_pairs": catalogs.pair_count,
        "train_cutoffs": [str(cutoff.date()) for cutoff in train_cutoffs],
        "valid_cutoff": str(valid_cutoff.date()),
        "cutoff_step_days": args.cutoff_step_days,
        "seed": 42,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lib_versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_checksums(model_dir)

    bundle = ModelBundle.load(model_dir)
    logger.info("End-to-end проверка на valid cutoff %s", valid_cutoff.date())
    metrics = evaluate_bundle(sales, stock, bundle, valid_cutoff)
    for row in metrics["grid"]:
        logger.info("  threshold=%s -> %s", row["threshold"], row)
    logger.info(
        "Выбранный порог: %s, метрики: %s",
        metrics["chosen_threshold"],
        metrics["chosen"],
    )

    config["postprocess"]["positive_threshold"] = metrics["chosen_threshold"]
    config["valid_metrics"] = metrics
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_checksums(model_dir)

    elapsed = time.perf_counter() - started
    logger.info("Артефакты сохранены в %s (%.1f мин)", model_dir.resolve(), elapsed / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

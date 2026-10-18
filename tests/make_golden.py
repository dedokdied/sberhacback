"""Готовит детерминированные тестовые CSV из train-датасета.

Скрипт берёт ``--pairs`` наиболее активных пар (склад×артикул) за всю историю и
выгружает их продажи и остатки как «как будто от организаторов» CSV для
contract/parity тестов. Файлы кладутся в ``tests/data/``.

Запуск (из корня forecast-container):

    python -m tests.make_golden \
        --sales ../csv/sales_dataset_hakaton_train.csv \
        --stock ../csv/stock_dataset_hakaton_train.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import data_loader

logger = logging.getLogger("make_golden")

SALES_COLUMN_ORDER = [
    "date",
    "composite_division",
    "cities_name",
    "warehouse_name",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "article",
    "characteristic",
    "quantity",
    "sales_sum",
    "discount_sum",
    "cost_price",
    "volume_weight",
]
STOCK_COLUMN_ORDER = ["date", "warehouse_name", "article", "stock"]
KEY = ["warehouse_name", "article"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация golden CSV для тестов")
    parser.add_argument("--sales", required=True)
    parser.add_argument("--stock", required=True)
    parser.add_argument("--out", default="tests/data")
    parser.add_argument("--pairs", type=int, default=35)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    sales = data_loader.load_sales(args.sales, strict=False)
    stock = data_loader.load_stock(args.stock, strict=False, resolve_conflicts="mean")

    counts = (
        sales.groupby(KEY, sort=False)
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    pairs = counts.head(args.pairs).index
    logger.info("Отобрано пар: %d", len(pairs))

    sales_index = pd.MultiIndex.from_frame(sales[KEY])
    stock_index = pd.MultiIndex.from_frame(stock[KEY])
    sales_slice = sales[sales_index.isin(pairs)].copy()
    stock_slice = stock[stock_index.isin(pairs)].copy()

    sales_columns = [col for col in SALES_COLUMN_ORDER if col in sales_slice.columns]
    sales_slice = sales_slice[sales_columns].sort_values(
        ["date", *KEY], kind="stable"
    )
    for column in ("sales_sum", "discount_sum", "cost_price"):
        if column in sales_slice.columns:
            sales_slice[column] = sales_slice[column].round(4)
    stock_slice = stock_slice[STOCK_COLUMN_ORDER].sort_values(
        ["date", *KEY], kind="stable"
    )
    stock_slice["stock"] = stock_slice["stock"].round(3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sales_path = out_dir / "sales_golden.csv"
    stock_path = out_dir / "stock_golden.csv"
    sales_slice.to_csv(sales_path, index=False)
    stock_slice.to_csv(stock_path, index=False)

    logger.info(
        "sales_golden.csv: %d строк (%s..%s), %d пар -> %s",
        len(sales_slice),
        sales_slice["date"].min().date(),
        sales_slice["date"].max().date(),
        sales_slice.groupby(KEY, sort=False).ngroups,
        sales_path.resolve(),
    )
    logger.info(
        "stock_golden.csv: %d строк (%s..%s) -> %s",
        len(stock_slice),
        stock_slice["date"].min().date(),
        stock_slice["date"].max().date(),
        stock_path.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

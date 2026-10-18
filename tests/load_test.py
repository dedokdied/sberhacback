"""Простой нагрузочный прогон POST /predict (BE-09): время по итерациям.

Использование:

    # против запущенного сервиса
    BASE_URL=http://localhost:8000 python -m tests.load_test --iterations 3

    # без сервиса (in-process TestClient)
    python -m tests.load_test --iterations 1 --sales tests/data/sales_golden.csv \
        --stock tests/data/stock_golden.csv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Нагрузочный прогон POST /predict")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sales", default="tests/data/sales_golden.csv")
    parser.add_argument("--stock", default="tests/data/stock_golden.csv")
    return parser.parse_args(argv)


def post_once(post_callable, sales_bytes: bytes, stock_bytes: bytes) -> float:
    started = time.perf_counter()
    response = post_callable(sales_bytes, stock_bytes)
    rows = len(response.content)
    elapsed = time.perf_counter() - started
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text[:300]}"
    print(f"  {elapsed:7.2f} c, ответ {rows / 1e6:6.1f} МБ")
    return elapsed


def main(argv=None) -> int:
    args = parse_args(argv)
    sales_bytes = Path(args.sales).read_bytes()
    stock_bytes = Path(args.stock).read_bytes()
    print(
        f"Вход: sales={args.sales} ({len(sales_bytes) / 1e6:.1f} МБ), "
        f"stock={args.stock} ({len(stock_bytes) / 1e6:.1f} МБ)"
    )

    base_url = os.environ.get("BASE_URL")
    if base_url:
        import httpx

        client = httpx.Client(base_url=base_url, timeout=3600)
        print(f"Режим: внешний сервис {base_url}")
        post = lambda sales, stock: client.post(  # noqa: E731
            "/predict",
            files={"sales": ("sales.csv", sales, "text/csv"), "stock": ("stock.csv", stock, "text/csv")},
        )
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        print("Режим: in-process TestClient")
        test_client = TestClient(app)
        test_client.__enter__()
        post = lambda sales, stock: test_client.post(  # noqa: E731
            "/predict",
            files={"sales": ("sales.csv", sales, "text/csv"), "stock": ("stock.csv", stock, "text/csv")},
        )

    try:
        times = [post_once(post, sales_bytes, stock_bytes) for _ in range(args.iterations)]
    finally:
        if not base_url:
            test_client.__exit__(None, None, None)

    print(
        f"Итого: min={min(times):.2f} c, max={max(times):.2f} c, "
        f"mean={sum(times) / len(times):.2f} c"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

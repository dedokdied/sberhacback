"""Общие фикстуры тестов: TestClient с загруженной моделью и golden-файлы."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = ROOT / "model"

GOLDEN_SALES = DATA_DIR / "sales_golden.csv"
GOLDEN_STOCK = DATA_DIR / "stock_golden.csv"

os.environ.setdefault("MODEL_DIR", str(MODEL_DIR))


def _require_artifacts() -> None:
    if not (MODEL_DIR / "lightgbm.txt").exists():
        pytest.skip(
            "Артефакты модели не найдены в model/: сначала выполните "
            "python -m src.train ... (см. README)"
        )


@pytest.fixture(scope="session")
def client():
    _require_artifacts()
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def golden_files() -> tuple[Path, Path]:
    if not GOLDEN_SALES.exists() or not GOLDEN_STOCK.exists():
        pytest.skip("Нет tests/data/*_golden.csv: запустите python -m tests.make_golden")
    return GOLDEN_SALES, GOLDEN_STOCK


@pytest.fixture(scope="session")
def golden_frames(golden_files) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_path, stock_path = golden_files
    return pd.read_csv(sales_path, dtype=str), pd.read_csv(stock_path, dtype=str)


@pytest.fixture(scope="session")
def posted(client, golden_files):
    """Один успешный POST на весь тестовый прогон."""
    sales_path, stock_path = golden_files
    with open(sales_path, "rb") as sales_file, open(stock_path, "rb") as stock_file:
        response = client.post(
            "/predict",
            files={
                "sales": ("sales_golden.csv", sales_file.read(), "text/csv"),
                "stock": ("stock_golden.csv", stock_file.read(), "text/csv"),
            },
        )
    assert response.status_code == 200, response.text
    return response


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def post_files(client, sales_bytes: bytes, stock_bytes: bytes):
    return client.post(
        "/predict",
        files={
            "sales": ("sales.csv", sales_bytes, "text/csv"),
            "stock": ("stock.csv", stock_bytes, "text/csv"),
        },
    )

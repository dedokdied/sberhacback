"""Parity-тесты offline / API: POST /predict должен совпадать с src.predict.predict_64.

Проверка по разделу 8.2 ТЗ: одинаковые входные CSV, сортировка по
(date, warehouse_name, article), rtol=0, atol=1e-7.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import data_loader
from src.predict import ModelBundle, predict_64
from tests.conftest import MODEL_DIR, frame_to_csv_bytes, post_files

KEYS = ["date", "warehouse_name", "article"]


@pytest.fixture(scope="session")
def bundle() -> ModelBundle:
    return ModelBundle.load(MODEL_DIR)


def offline_predict(bundle: ModelBundle, sales_bytes: bytes, stock_bytes: bytes) -> pd.DataFrame:
    sales = data_loader.load_sales(sales_bytes, catalogs=bundle.catalogs, strict=True)
    stock = data_loader.load_stock(stock_bytes, catalogs=bundle.catalogs, strict=True)
    return predict_64(sales, stock, bundle)


def assert_parity(api_bytes: bytes, reference: pd.DataFrame) -> None:
    api = pd.read_csv(io.BytesIO(api_bytes)).sort_values(KEYS, kind="stable").reset_index(drop=True)
    ref = reference.sort_values(KEYS, kind="stable").reset_index(drop=True)
    assert len(api) == len(ref)
    for column in KEYS:
        assert (api[column].to_numpy() == ref[column].to_numpy()).all(), column
    assert np.allclose(
        api["quantity"].to_numpy(dtype=np.float64),
        ref["quantity"].to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-7,
    )


def test_parity_golden(client, bundle, golden_files, posted):
    sales_path, stock_path = golden_files
    sales_bytes, stock_bytes = sales_path.read_bytes(), stock_path.read_bytes()
    assert_parity(posted.content, offline_predict(bundle, sales_bytes, stock_bytes))


def test_parity_sparse_stock(client, bundle, golden_frames):
    sales, stock = golden_frames
    sparse_stock = stock.iloc[::3]
    sales_bytes = frame_to_csv_bytes(sales)
    stock_bytes = frame_to_csv_bytes(sparse_stock)
    response = post_files(client, sales_bytes, stock_bytes)
    assert response.status_code == 200, response.text
    assert_parity(response.content, offline_predict(bundle, sales_bytes, stock_bytes))


def test_parity_shifted_cutoff(client, bundle, golden_frames):
    sales, stock = golden_frames
    dates = pd.to_datetime(sales["date"])
    shifted = sales[dates <= (dates.max() - pd.Timedelta(days=30))]
    sales_bytes = frame_to_csv_bytes(shifted)
    stock_bytes = frame_to_csv_bytes(stock)
    response = post_files(client, sales_bytes, stock_bytes)
    assert response.status_code == 200, response.text
    assert_parity(response.content, offline_predict(bundle, sales_bytes, stock_bytes))

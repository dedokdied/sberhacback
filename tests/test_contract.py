"""Контрактные и негативные тесты HTTP API (таблица 15 backend-ТЗ)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import MODEL_DIR, frame_to_csv_bytes, post_files

EXPECTED_ROWS = 562_496
EXPECTED_DATES = 64


def _error(response, expected_status: int, expected_code: str) -> dict:
    assert response.status_code == expected_status, response.text
    payload = response.json()
    assert set(payload) == {"detail", "code"}, payload
    assert payload["code"] == expected_code, payload
    assert isinstance(payload["detail"], str) and payload["detail"], payload
    return payload


def _catalogs() -> dict:
    return json.loads((MODEL_DIR / "catalogs.json").read_text(encoding="utf-8"))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert isinstance(payload["model_version"], str)


def test_get_before_post_returns_409(golden_files):
    from app.main import app

    with TestClient(app) as fresh_client:
        response = fresh_client.get("/predict", params={"article": _catalogs()["articles"][0]})
    _error(response, 409, "CONFLICT")


def test_post_builds_full_grid(posted, client):
    frame = pd.read_csv(io.BytesIO(posted.content))
    assert list(frame.columns) == ["date", "warehouse_name", "article", "quantity"]
    assert len(frame) == EXPECTED_ROWS
    assert frame["date"].nunique() == EXPECTED_DATES
    assert not frame.duplicated(subset=["date", "warehouse_name", "article"]).any()
    assert np.isfinite(frame["quantity"]).all()
    assert (frame["quantity"] >= 0).all()

    catalogs = _catalogs()
    assert frame["warehouse_name"].nunique() == len(catalogs["warehouses"]) == 11
    assert frame["article"].nunique() == len(catalogs["articles"]) == 799

    dates = pd.to_datetime(frame["date"].unique())
    span = pd.date_range(dates.min(), periods=EXPECTED_DATES, freq="D")
    assert list(dates) == list(span)

    assert posted.headers["x-forecast-rows"] == str(EXPECTED_ROWS)
    assert posted.headers["content-type"].startswith("text/csv")


def test_post_is_deterministic(client, golden_files, posted):
    sales_path, stock_path = golden_files
    response = post_files(
        client, sales_path.read_bytes(), stock_path.read_bytes()
    )
    assert response.status_code == 200
    assert response.content == posted.content


def test_get_article_filter(posted, client):
    catalogs = _catalogs()
    article = catalogs["articles"][0]
    response = client.get("/predict", params={"article": article})
    assert response.status_code == 200
    frame = pd.read_csv(io.BytesIO(response.content))
    assert len(frame) == EXPECTED_DATES * len(catalogs["warehouses"]) == 704
    assert set(frame["article"]) == {article}


def test_get_warehouse_filter(posted, client):
    catalogs = _catalogs()
    warehouse = catalogs["warehouses"][0]
    response = client.get("/predict", params={"warehouse": warehouse})
    assert response.status_code == 200
    frame = pd.read_csv(io.BytesIO(response.content))
    assert len(frame) == EXPECTED_DATES * len(catalogs["articles"]) == 51_136
    assert set(frame["warehouse_name"]) == {warehouse}


def test_get_pair_filter(posted, client):
    catalogs = _catalogs()
    response = client.get(
        "/predict",
        params={"article": catalogs["articles"][5], "warehouse": catalogs["warehouses"][2]},
    )
    assert response.status_code == 200
    frame = pd.read_csv(io.BytesIO(response.content))
    assert len(frame) == EXPECTED_DATES == 64


def test_get_without_params_returns_400(posted, client):
    _error(client.get("/predict"), 400, "BAD_REQUEST")


def test_get_unknown_id_returns_404(posted, client):
    _error(
        client.get("/predict", params={"article": "char_article_99999"}),
        404,
        "NOT_FOUND",
    )
    _error(
        client.get("/predict", params={"warehouse": "warehouse_name_99999"}),
        404,
        "NOT_FOUND",
    )


def test_all_csv(posted, client):
    response = client.get("/predict/all", params={"format": "csv"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    text = response.content.decode("utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "date,warehouse_name,article,quantity"
    assert len(lines) == EXPECTED_ROWS + 1


def test_all_json_streamed(posted, client):
    response = client.get("/predict/all", params={"format": "json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.content
    assert body.startswith(b"[") and body.endswith(b"]")
    assert body.count(b'"date":') == EXPECTED_ROWS
    start = body.index(b"{")
    first = json.loads(body[start : body.index(b"}") + 1])
    assert set(first) == {"date", "warehouse_name", "article", "quantity"}


def test_all_bad_format(posted, client):
    _error(client.get("/predict/all", params={"format": "xml"}), 400, "BAD_REQUEST")


def test_empty_csv_returns_400(client, golden_frames):
    _, stock = golden_frames
    response = post_files(client, b"", frame_to_csv_bytes(stock))
    _error(response, 400, "BAD_REQUEST")


def test_broken_csv_returns_400(client, golden_frames):
    _, stock = golden_frames
    broken = b'date,warehouse_name,"article\n2024-01-01,sales,warehouse_name_0\n'
    response = post_files(client, broken, frame_to_csv_bytes(stock))
    _error(response, 400, "BAD_REQUEST")


def test_missing_column_returns_422(client, golden_frames):
    sales, stock = golden_frames
    broken = sales.drop(columns=["quantity"])
    response = post_files(client, frame_to_csv_bytes(broken), frame_to_csv_bytes(stock))
    _error(response, 422, "VALIDATION_ERROR")


def test_missing_stock_file_returns_422(client, golden_files):
    sales_path, _ = golden_files
    response = client.post(
        "/predict", files={"sales": ("sales.csv", sales_path.read_bytes(), "text/csv")}
    )
    _error(response, 422, "VALIDATION_ERROR")


def test_invalid_date_returns_422(client, golden_frames):
    sales, stock = golden_frames
    broken = sales.copy()
    broken.loc[0, "date"] = "2026-13-01"
    response = post_files(client, frame_to_csv_bytes(broken), frame_to_csv_bytes(stock))
    _error(response, 422, "VALIDATION_ERROR")


def test_negative_quantity_returns_422(client, golden_frames):
    sales, stock = golden_frames
    broken = sales.copy()
    broken.loc[0, "quantity"] = "-1"
    response = post_files(client, frame_to_csv_bytes(broken), frame_to_csv_bytes(stock))
    _error(response, 422, "VALIDATION_ERROR")


def test_non_numeric_quantity_returns_422(client, golden_frames):
    sales, stock = golden_frames
    broken = sales.copy()
    broken.loc[1, "quantity"] = "abc"
    response = post_files(client, frame_to_csv_bytes(broken), frame_to_csv_bytes(stock))
    _error(response, 422, "VALIDATION_ERROR")


def test_negative_stock_accepted_and_clipped(client, golden_frames):
    """Отрицательный stock не отклоняется (п. 10.4 отключён по согласованию) — прогноз отдаётся."""
    sales, stock = golden_frames
    broken = stock.copy()
    broken.loc[0, "stock"] = "-2"
    response = post_files(client, frame_to_csv_bytes(sales), frame_to_csv_bytes(broken))
    assert response.status_code == 200, response.text
    lines = response.content.decode("utf-8").strip().splitlines()
    assert len(lines) == EXPECTED_ROWS + 1


def test_short_history_returns_422(client, golden_frames):
    sales, stock = golden_frames
    dates = pd.to_datetime(sales["date"])
    last = dates.max()
    truncated = sales[dates >= (last - pd.Timedelta(days=178))]
    response = post_files(client, frame_to_csv_bytes(truncated), frame_to_csv_bytes(stock))
    payload = _error(response, 422, "VALIDATION_ERROR")
    assert "истори" in payload["detail"].lower()


def test_unknown_ids_return_422(client, golden_frames):
    sales, stock = golden_frames
    broken = sales.copy()
    broken.loc[0, "article"] = "char_article_99999"
    response = post_files(client, frame_to_csv_bytes(broken), frame_to_csv_bytes(stock))
    payload = _error(response, 422, "VALIDATION_ERROR")
    assert "каталог" in payload["detail"].lower()


def test_conflicting_stock_duplicates_return_422(client, golden_frames):
    sales, stock = golden_frames
    row = stock.iloc[[0]].copy()
    duplicated = pd.concat([stock, row], ignore_index=True)
    duplicated.loc[len(duplicated) - 1, "stock"] = str(float(row["stock"].iloc[0]) + 5.0)
    response = post_files(client, frame_to_csv_bytes(sales), frame_to_csv_bytes(duplicated))
    payload = _error(response, 422, "VALIDATION_ERROR")
    assert "конфликт" in payload["detail"].lower()

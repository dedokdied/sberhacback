# Протокол замеров времени и памяти (BE-09)

Дата: 2026-10-17. Среда разработки: Windows 10, CPU Intel i7-8750H (6 ядер / 12 потоков),
16 ГБ RAM, Python 3.13 (venv), LightGBM 4.7.0. Локальный запуск Uvicorn
(`--workers 1`, порт 8100). Docker в среде разработки недоступен — контейнерные
замеры выполнить в Docker-окружении заказчика (команды ниже).

## 1. POST /predict — полный инференс (562 496 строк)

| Вход | Размер входа | В ответе | Время | RSS сервиса* |
|---|---|---|---|---|
| golden (35 пар, ~2,5 года) | sales 4,4 МБ, stock 1,6 МБ | 562 496 строк (27,5 МБ) | **7,3–7,5 с** | 165 МБ |
| полный срез последних 180 дней (все пары) | sales 46,5 МБ (271 612 строк), stock 71,0 МБ (1 408 656 строк) | 562 496 строк (31,8 МБ) | **24,8 с** | 171 МБ |

\* Working Set процесса сервиса после завершения запроса (включая кэш последнего
прогноза). Пиковое потребление в момент валидации CSV — выше, но с большим запасом
ниже лимита 64 ГБ.

Лимиты ТЗ: POST ≤ 60 минут, RAM ≤ 64 ГБ, GET ≤ 10 минут. Запас: >140x по времени.

## 2. GET /predict и /predict/all

| Запрос | Строк | Наблюдение |
|---|---|---|
| GET /predict?article=... (11 складов) | 704 | отдача стримингом, < 1 с |
| GET /predict?warehouse=... (799 артикулов) | 51 136 | стримингом, < 1 с |
| GET /predict?article=..&warehouse=.. | 64 | стримингом, < 1 с |
| GET /predict/all?format=csv | 562 497 строк файла | чанки по 20 000 строк |
| GET /predict/all?format=json | 562 496 объектов | чанки по 20 000, без полного to_dict |

## 3. Тесты

| Прогон | Результат | Время |
|---|---|---|
| `pytest tests` (26 тестов: контракт, негативные, parity, детерминизм) | 26 passed | 66 с |
| `tests/test_api.sh` против живого Uvicorn | 17/17 OK | ~40 с |

## 4. Как воспроизвести

```bash
# Локально
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
BASE_URL=http://127.0.0.1:8000 python -m tests.load_test --iterations 3
BASE_URL=http://127.0.0.1:8000 bash tests/test_api.sh

# Полный вход (последние 180 дней)
python -m tests.load_test --iterations 1 \
  --sales /path/sales_last_180.csv --stock /path/stock_last_180.csv

# Docker (замер RAM внутри контейнера)
docker build -t nonton-forecast:latest .
docker run --rm -d --name nonton-forecast -p 8000:8000 --memory=32g --cpus=8 nonton-forecast:latest
docker stats nonton-forecast
```

## 5. Модель

- best_iteration = 125 (early stopping 100), обучение полного набора из 16 cutoff-ов:
  8 987 648 строк × 27 признаков — **6,2 мин** на указанной машине.
- Артефакты: model/lightgbm.txt (1,4 МБ), config.json, catalogs.json, checksums.json.
- Качество на отложенном cutoff 2026-05-01 (64 дня, официальные метрики кейса):
  SMAPE = 158,10 %, WAPE = 90,53 %, RMSE = 0,8165, Bias = −7,18 %, RMSLE = 0,3086
  (порог обнуления слабых плюсов подобран на этом же окне, см. config.json).

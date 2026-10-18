# Nonton Forecast API — сервис прогнозирования продаж (склады × артикулы, 64 дня)

Автономный FastAPI-сервис: принимает два CSV (продажи и остатки), считает прогноз
**562 496 строк** (11 складов × 799 артикулов × 64 дня) предобученной LightGBM-моделью
и отдаёт его стримингом в CSV/JSON. Обучения внутри API-запроса нет, сеть на инференсе
не используется.

Архитектура ML: **Horizon-Aware Global Direct LightGBM** — единая модель получает
исторические признаки пары (окно 180 дней), номер горизонта `h=1..64` и календарные
признаки; собственные прогнозы как лаги не используются.

## Состав репозитория

```
forecast-container/
├── app/                    # HTTP-слой (FastAPI)
│   ├── main.py             # маршруты /health /predict /predict/all, единые ошибки
│   ├── predictor.py        # ForecastService: кэш, 409-lock, стриминг CSV/JSON
│   └── validators.py       # строгие проверки CSV, маппинг ошибок в 400/422/404/409
├── src/                    # ML-ядро (общее для offline и API)
│   ├── data_loader.py      # парсинг/агрегация/валидация CSV
│   ├── features.py         # единая генерация признаков offline+API
│   ├── predict.py          # ModelBundle и predict_64 — единая точка инференса
│   └── train.py            # обучение и сборка артефактов model/ (offline)
├── model/                  # lightgbm.txt, config.json, catalogs.json, checksums.json
├── tests/                  # pytest (контракт, негативные, parity), test_api.sh, load_test.py
│   └── data/               # golden-фикстуры (sales/stock), см. tests/make_golden.py
├── reports/benchmark.md    # протокол замеров времени/памяти
├── Dockerfile, .dockerignore
├── requirements.txt, requirements-dev.txt, requirements.lock
└── README.md
```

## Быстрый старт

### Docker (целевой способ)

```bash
docker build -t nonton-forecast:latest .
docker run --rm --name nonton-forecast -p 8000:8000 --memory=32g --cpus=8 nonton-forecast:latest
curl -f http://localhost:8000/health
```

Образ собирается из `python:3.11-slim` + `libgomp1`; train-датасет в образ не копируется
(`.dockerignore`), модель загружается из `model/` на startup — при отсутствии артефактов
контейнер не стартует. Многопроцессность не используется (`--workers 1`), кэш хранится в RAM.

### Локально без Docker

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt        # Windows; Linux: .venv/bin/pip
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

## API-паспорт

| Метод | Путь | Вход | Ответ |
|---|---|---|---|
| POST | `/predict` | multipart `sales` + `stock` (оба обязательны) | `text/csv`: 562 496 строк + заголовок |
| GET | `/predict` | query `article` и/или `warehouse` (хотя бы один) | `text/csv` с фильтром |
| GET | `/predict/all` | `format=csv` (по умолчанию) или `json` | полный прогноз стримингом |
| GET | `/health` | — | `{"status":"ok","model_loaded":true,"model_version":"1.0.0"}` |

Примеры:

```bash
curl -X POST http://localhost:8000/predict \
  -F "sales=@sales_last_180.csv" -F "stock=@stock_last_180.csv" -o forecast.csv

curl 'http://localhost:8000/predict?article=char_article_1'
curl 'http://localhost:8000/predict?warehouse=warehouse_name_3'
curl 'http://localhost:8000/predict?article=char_article_1&warehouse=warehouse_name_3'
curl 'http://localhost:8000/predict/all?format=csv' -o full_forecast.csv
curl 'http://localhost:8000/predict/all?format=json' -o full_forecast.json
```

### Формат ошибок (единый)

```json
{"detail": "Недостаточно истории: 179 дней, минимум 180", "code": "VALIDATION_ERROR"}
```

| Код HTTP | code | Когда |
|---|---|---|
| 400 | `BAD_REQUEST` | пустой/сломанный CSV; GET без параметров; неизвестный `format` |
| 422 | `VALIDATION_ERROR` | схема/типы/даты/история/дубли-конфликты stock/ID вне каталога; отсутствует файл формы |
| 404 | `NOT_FOUND` | неизвестный `article`/`warehouse` в GET-фильтре |
| 409 | `CONFLICT` | GET до первого успешного POST; параллельный POST во время расчёта |

### Кэш и конкурентность (решения V1)

- Прогноз вычисляется только при POST; после успеха кэш в RAM заменяется **атомарно**.
- При ошибке POST предыдущий успешный прогноз остаётся доступным.
- Параллельный POST не ждёт очередь — получает **409** (выбор для хакатона).
- **После рестарта процесса кэш пуст** — нужен новый POST (in-memory, без внешнего хранилища).
- Повторный POST с теми же байтами даёт идентичный результат (детерминированные
  параметры LightGBM, фиксированные версии, фиксированный порядок ключей/признаков).
- Строгий каталог `model/catalogs.json` (11×799): неизвестные ID на входе — 422 (POST),
  неизвестный фильтр — 404 (GET); динамическое расширение каталога — вне MVP.

## Правила валидации входных CSV

- **sales**: обязательны `date, warehouse_name, article, quantity`; дубли
  `(date, warehouse_name, article)` (разные департаменты) агрегируются `quantity=sum`;
  денежные колонки опциональны (пустые/отрицательные допустимы, агрегируются sum/mean).
- **stock**: обязательны `date, warehouse_name, article|char_article, stock`;
  конфликтующие дубли `(date, warehouse_name, article)` → **422** (политика не согласована),
  идентичные дубли дедуплицируются; отрицательный остаток → 422.
- История: `max(date) − min(date) + 1 ≥ 180` календарных дней (разрывы покрытия логируются).
- Остатки после `max(sales.date)` в признаки не попадают; дата старта прогноза —
  строго `max(sales.date) + 1`.
- CSV читаются как строки (ведущие нули ID сохраняются); числа — только для
  `quantity`/`stock` и денежных колонок.

## Модель

Артефакты в `model/`: `lightgbm.txt` + `config.json` (`model_version`, `schema_version`,
`features[]`, `history_days`, `horizon`, `lgb_params`, `postprocess`) + `catalogs.json`
(склады/артикулы/категории) + `checksums.json` (SHA-256, проверяется на startup).

Целевая переменная — Tweedie по сырому `quantity` (zero-inflated спрос); постобработка:
`clip ≥ 0`, обнуление «спящих» серий (нет продаж в окне 180 дней) и порог обнуления
слабых плюсов, подобранный по официальным метрикам кейса на отложенном окне (см.
`config.json → postprocess.positive_threshold`).

Контрольные метрики на отложенном cutoff 2026-05-01 (64 дня): SMAPE = 158,1 %,
WAPE = 90,5 %, RMSE = 0,817, Bias = −7,2 %, RMSLE = 0,309. Полный протокол замеров —
`reports/benchmark.md`.

Переобучение (offline, вне Docker):

```bash
python -m src.train \
  --sales ../csv/sales_dataset_hakaton_train.csv \
  --stock ../csv/stock_dataset_hakaton_train.csv \
  --out model --objective tweedie
```

Скрипт сам подбирает порог постобработки на последнем cutoff и перезаписывает
`config.json` + `checksums.json`.

## Тесты

```bash
python -m tests.make_golden \        # при необходимости пересобрать tests/data/*.csv
  --sales ../csv/sales_dataset_hakaton_train.csv \
  --stock ../csv/stock_dataset_hakaton_train.csv

python -m pytest tests -q            # 26 тестов: контракт, негативные, parity, детерминизм
BASE_URL=http://localhost:8000 bash tests/test_api.sh   # 17 проверок через curl
BASE_URL=http://localhost:8000 python -m tests.load_test --iterations 3
```

Parity offline/API (раздел 8.2 ТЗ): `tests/test_parity.py` сравнивает POST-ответ с
`src.predict.predict_64` на тех же CSV (сортировка по `date, warehouse_name, article`,
`rtol=0, atol=1e-7`) для трёх сценариев: golden, разреженные остатки, сдвинутый cutoff.

## Известные ограничения

- Docker-сборка проверена только «на бумаге»: в среде разработки Docker недоступен —
  соберите образ в целевом окружении (команды выше).
- Кэш не переживает рестарт процесса (решение V1); при нескольких воркерах кэш был бы
  рассинхронизирован, поэтому зафиксирован `--workers 1`.
- Строгий каталог: новые ID требуют отдельной версии API (переобучение + новые каталоги).
- Подбор порога постобработки выполнен на одном отложенном окне — при переобучении
  пересчитывается автоматически `src/train.py`.

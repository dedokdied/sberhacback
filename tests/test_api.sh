#!/usr/bin/env bash
# Сквозная проверка контракта сервиса через curl.
# Требует запущенный сервис (uvicorn или Docker) и golden-файлы tests/data/.
#
# Использование:
#   BASE_URL=http://localhost:8000 tests/test_api.sh
#
# Выход: 0 — все проверки пройдены, 1 — есть ошибка.

set -u

BASE_URL="${BASE_URL:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
SALES="$DATA_DIR/sales_golden.csv"
STOCK="$DATA_DIR/stock_golden.csv"
TMP_DIR=".api_test_tmp_$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

FAILURES=0
PASS_COUNT=0

ok() { echo "OK   $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
bad() { echo "FAIL $1"; FAILURES=$((FAILURES + 1)); }

require() {
  local name="$1"
  shift
  if "$@"; then ok "$name"; else bad "$name"; fi
}

http_code() {
  curl -s -o /dev/null -w "%{http_code}" "$@"
}

echo "== Сервис: $BASE_URL =="

# --- health ---
HEALTH_CODE=$(http_code "$BASE_URL/health")
[ "$HEALTH_CODE" = "200" ] && ok "GET /health -> 200" || bad "GET /health -> $HEALTH_CODE"
curl -s "$BASE_URL/health" | grep -q '"model_loaded": *true' && ok "health.model_loaded" || bad "health.model_loaded"

# --- POST /predict (полный расчёт) ---
POST_CODE=$(curl -s -X POST "$BASE_URL/predict" \
  -F "sales=@$SALES" -F "stock=@$STOCK" -o "$TMP_DIR/forecast.csv" -w "%{http_code}")
[ "$POST_CODE" = "200" ] && ok "POST /predict -> 200" || bad "POST /predict -> $POST_CODE"

LINES=$(wc -l < "$TMP_DIR/forecast.csv" | tr -d ' ')
[ "$LINES" = "562497" ] && ok "CSV: 562 496 строк + заголовок" || bad "CSV строк: $LINES (ожидалось 562497)"

head -n 1 "$TMP_DIR/forecast.csv" | grep -q '^date,warehouse_name,article,quantity$' \
  && ok "CSV заголовок" || bad "CSV заголовок"

# --- GET фильтры ---
curl -s "$BASE_URL/predict?article=char_article_0" -o "$TMP_DIR/pair.csv"
PAIR_LINES=$(($(wc -l < "$TMP_DIR/pair.csv" | tr -d ' ') - 1))
[ "$PAIR_LINES" = "704" ] && ok "GET article -> 704 строки (11x64)" || bad "GET article -> $PAIR_LINES"

curl -s "$BASE_URL/predict?warehouse=warehouse_name_0" -o "$TMP_DIR/wh.csv"
WH_LINES=$(($(wc -l < "$TMP_DIR/wh.csv" | tr -d ' ') - 1))
[ "$WH_LINES" = "51136" ] && ok "GET warehouse -> 51 136 строк (799x64)" || bad "GET warehouse -> $WH_LINES"

curl -s "$BASE_URL/predict?article=char_article_0&warehouse=warehouse_name_0" -o "$TMP_DIR/one.csv"
ONE_LINES=$(($(wc -l < "$TMP_DIR/one.csv" | tr -d ' ') - 1))
[ "$ONE_LINES" = "64" ] && ok "GET пара -> 64 строки" || bad "GET пара -> $ONE_LINES"

# --- GET /predict/all ---
ALL_CODE=$(curl -s "$BASE_URL/predict/all?format=csv" -o "$TMP_DIR/all.csv" -w "%{http_code}")
ALL_LINES=$(wc -l < "$TMP_DIR/all.csv" | tr -d ' ')
[ "$ALL_CODE" = "200" ] && [ "$ALL_LINES" = "562497" ] && ok "GET /predict/all?format=csv" || bad "GET all csv: $ALL_CODE, строк $ALL_LINES"

JSON_CODE=$(curl -s "$BASE_URL/predict/all?format=json" -o "$TMP_DIR/all.json" -w "%{http_code}")
[ "$JSON_CODE" = "200" ] && python -c "import json,sys; data=json.load(open(sys.argv[1])); assert len(data)==562496" "$TMP_DIR/all.json" 2>/dev/null \
  && ok "GET /predict/all?format=json" || bad "GET all json"

# --- негативные сценарии ---
CODE=$(http_code "$BASE_URL/predict")
ERROR_CODE=$(curl -s "$BASE_URL/predict" | grep -o '"code": *"[A-Z_]*"')
[ "$CODE" = "400" ] && echo "$ERROR_CODE" | grep -q BAD_REQUEST && ok "GET без параметров -> 400 BAD_REQUEST" || bad "GET без параметров -> $CODE"
[ "$(http_code "$BASE_URL/predict/all?format=xml")" = "400" ] && ok "format=xml -> 400" || bad "format=xml"
[ "$(http_code "$BASE_URL/predict?article=char_article_99999")" = "404" ] && ok "неизвестный article -> 404" || bad "неизвестный article"

: > "$TMP_DIR/empty.csv"
[ "$(http_code -X POST "$BASE_URL/predict" -F "sales=@$TMP_DIR/empty.csv" -F "stock=@$STOCK")" = "400" ] \
  && ok "пустой CSV -> 400" || bad "пустой CSV"

printf 'date,warehouse_name,article\n2026-01-01,w,a\n' > "$TMP_DIR/nocol.csv"
[ "$(http_code -X POST "$BASE_URL/predict" -F "sales=@$TMP_DIR/nocol.csv" -F "stock=@$STOCK")" = "422" ] \
  && ok "нет колонки quantity -> 422" || bad "нет колонки quantity"

printf 'date,warehouse_name,article,quantity\n2026-13-99,w,a,1\n' > "$TMP_DIR/baddate.csv"
[ "$(http_code -X POST "$BASE_URL/predict" -F "sales=@$TMP_DIR/baddate.csv" -F "stock=@$STOCK")" = "422" ] \
  && ok "невалидная дата -> 422" || bad "невалидная дата"

[ "$(http_code -X POST "$BASE_URL/predict" -F "sales=@$SALES")" = "422" ] \
  && ok "отсутствует файл stock -> 422" || bad "отсутствует файл stock"

echo
echo "Пройдено: $PASS_COUNT, ошибок: $FAILURES"
exit "$FAILURES"

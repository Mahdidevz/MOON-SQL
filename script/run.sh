#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH"

tables="$ROOT_DIR/data/spider_data/tables.json"
db_path="$ROOT_DIR/data/spider_data/database"
PROCESS_NUM=1
API_CALL_NUM=2

short_model_name='llama-3.3-70b-versatile'
long_model_name='llama-3.3-70b-versatile'

# ── Dynamic path overrides ─────────────────────────────────────────────────
# These variables can be set externally (e.g. by test_pipeline.sh) to switch
# between the full dataset and a smaller subset without touching core logic.
# Defaults preserve the original behaviour.
dev_path="${DEV_PATH:-$ROOT_DIR/data/spider/dev_50.json}"
data_mode="${DATA_MODE:-dev_50}"
# ──────────────────────────────────────────────────────────────────────────

first_output_path="$ROOT_DIR/data/intermediate_datasets/first_round_test.sql"
third_output_path="$ROOT_DIR/data/intermediate_datasets/third_round.sql"
processed_dataset_path="$ROOT_DIR/data/generate_datasets/preprocessed_data.json"
final_output_path="$ROOT_DIR/data/intermediate_datasets/predict_${data_mode}.json"

RETRY_NUM=10

directory="$ROOT_DIR/data/intermediate_datasets"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

directory="$ROOT_DIR/data/generate_datasets"
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

current_time=$(date)
echo $current_time

echo "preprocessing..."
cd "$ROOT_DIR"

python src/preprocess/preprocessing.py \
    --mode "test" \
    --table_path "$tables" \
    --input_dataset_path "$dev_path" \
    --output_dataset_path "$processed_dataset_path" \
    --db_path "$db_path" \
    --target_type "sql" \
    --process_num "$PROCESS_NUM"

echo "first round..."
python src/pipeline/first_module.py \
    --dev_path "$dev_path" \
    --data_path "$processed_dataset_path" \
    --output_path "$first_output_path" \
    --short_model_name "$short_model_name" \
    --long_model_name "$long_model_name" \
    --process_num "$API_CALL_NUM"

echo "third round..."
python src/pipeline/third_module.py \
    --dev_path "$dev_path" \
    --data_path "$processed_dataset_path" \
    --input_path "$first_output_path" \
    --output_path "$third_output_path" \
    --db_path "$db_path" \
    --retry_num "$RETRY_NUM" \
    --short_model_name "$short_model_name" \
    --long_model_name "$long_model_name" \
    --process_num "$API_CALL_NUM"

echo "generate standard output..."
python src/utils/append_db_id.py \
    --dev_path "$dev_path" \
    --input_path "$third_output_path" \
    --output_path "$final_output_path"

current_time=$(date)
echo $current_time

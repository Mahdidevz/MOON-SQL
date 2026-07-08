#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

db_root_path="$PROJECT_ROOT/data/spider_data/database/"
predicted_sql_path="$PROJECT_ROOT/data/intermediate_datasets/"

# ── Dynamic path overrides ─────────────────────────────────────────────────
# Set externally (e.g. by test_pipeline.sh) to switch between datasets.
# Defaults preserve the original behaviour.
data_mode="${DATA_MODE:-dev_50}"
diff_json_path="${DEV_PATH:-$PROJECT_ROOT/data/spider/dev_50.json}"
ground_truth_path="$PROJECT_ROOT/data/spider_data/"
# ──────────────────────────────────────────────────────────────────────────
num_cpus=4
meta_time_out=30.0
mode_gt="gt"
mode_predict="gpt"

cd "$PROJECT_ROOT"

python src/evaluate/evaluation.py \
  --db_root_path "$db_root_path" \
  --predicted_sql_path "$predicted_sql_path" \
  --data_mode "$data_mode" \
  --ground_truth_path "$ground_truth_path" \
  --num_cpus "$num_cpus" \
  --mode_gt "$mode_gt" \
  --mode_predict "$mode_predict" \
  --diff_json_path "$diff_json_path" \
  --meta_time_out "$meta_time_out"

python src/evaluate/evaluation_ves.py \
  --db_root_path "$db_root_path" \
  --predicted_sql_path "$predicted_sql_path" \
  --data_mode "$data_mode" \
  --ground_truth_path "$ground_truth_path" \
  --num_cpus "$num_cpus" \
  --mode_gt "$mode_gt" \
  --mode_predict "$mode_predict" \
  --diff_json_path "$diff_json_path" \
  --meta_time_out "$meta_time_out"

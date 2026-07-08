#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

db_root_path="$PROJECT_ROOT/data/spider_data/database/"
diff_json_path="$PROJECT_ROOT/data/spider/dev_50.json"
ground_truth_path="$PROJECT_ROOT/data/spider_data/"
predicted_sql_path="$PROJECT_ROOT/data/intermediate_datasets/"
data_mode="dev_50"
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

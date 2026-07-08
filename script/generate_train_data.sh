#!/bin/bash

# --- Groq API Config ---
export OPENAI_API_KEY="YOUR_GROQ_API_KEY"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
# -----------------------

tables="../bird/llm/data/train/train_tables.json"
dev_path="../bird/llm/data/train/train.json"
db_root_path='../bird/llm/data/databases/'
diff_json_path="../bird/llm/data/train/train.json"
ground_truth_path='../bird/llm/data/train/train_gold.sql'

short_model_name='llama-3.3-70b-versatile'
long_model_name='llama-3.3-70b-versatile'

output_path='./train_data/train_output.json'
first_output_path="./train_data/pre.sql"
processed_dataset_path='./train_data/preprocessed.json'
num_cpus=100
meta_time_out=30.0
mode_gt='gt'
mode_predict='generate'
PROCESS_NUM=1
API_CALL_NUM=100

echo 'start to preprocess data...'
python src/preprocess/preprocessing.py \
    --mode "test" \
    --table_path $tables \
    --input_dataset_path $dev_path \
    --output_dataset_path $processed_dataset_path \
    --db_path $db_root_path \
    --target_type "sql" \
    --process_num $PROCESS_NUM

echo 'start to generate pre-sql...'
python src/pipeline/first_module.py \
    --dev_path $dev_path \
    --data_path $processed_dataset_path \
    --output_path $first_output_path \
    --short_model_name $short_model_name \
    --long_model_name $long_model_name \
    --process_num $API_CALL_NUM

rm -f $output_path

echo 'start to generate train data...'
python3 -u src/preprocess/generate_train_data/record_error.py \
    --predicted_sql_path ${first_output_path} \
    --ground_truth_path ${ground_truth_path} \
    --db_root_path ${db_root_path} \
    --num_cpus ${num_cpus} \
    --meta_time_out ${meta_time_out} \
    --mode_gt ${mode_gt} \
    --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} \
    --output_path ${output_path} \
    --preprocessed_path ${processed_dataset_path} \
    --data_path ${dev_path}
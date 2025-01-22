#!/bin/bash

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_instruction_json" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_train.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_train.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_instruction_json" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_dev.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_dev.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_instruction_json" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_test.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_test.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_instruction_json" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_bc8_test.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_bc8_test.jsonl"

cat "datasets/bioredirect/bioredirect_train.jsonl" "datasets/bioredirect/bioredirect_dev.jsonl" > "datasets/bioredirect/bioredirect_train_dev.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_zs_instruction_jsonl" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_bc8_test.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_bc8_test.zs.jsonl" \

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_biored_zs_instruction_jsonl" \
    --in_pubtator_file "datasets/bioredirect/bioredirect_test.pubtator" \
    --out_json_file "datasets/bioredirect/bioredirect_test.zs.jsonl" \



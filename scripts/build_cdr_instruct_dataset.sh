#!/bin/bash

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_cdr_instruction_json" \
    --in_pubtator_file "datasets/cdr/CDR_TrainingSet.PubTator.txt" \
    --out_json_file "datasets/cdr/processed/train.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_cdr_instruction_json" \
    --in_pubtator_file "datasets/cdr/CDR_DevelopmentSet.PubTator.txt" \
    --out_json_file "datasets/cdr/processed/dev.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_cdr_instruction_json" \
    --in_pubtator_file "datasets/cdr/CDR_TestSet.PubTator.txt" \
    --out_json_file "datasets/cdr/processed/test.jsonl"

cat "datasets/cdr/processed/train.jsonl" "datasets/cdr/processed/dev.jsonl" > "datasets/cdr/processed/train_dev.jsonl"

python src/dataset_format_converter/convert_pubtator_2_json.py \
    --exp_option "gen_cdr_zs_instruction_jsonl" \
    --in_pubtator_file "datasets/cdr/CDR_TestSet.PubTator.txt" \
    --out_json_file "datasets/cdr/processed/test.zs.jsonl"

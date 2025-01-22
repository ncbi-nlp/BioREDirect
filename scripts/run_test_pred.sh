#!/bin/bash

in_pubtator_file="datasets/bioredirect/bioredirect_bc8_test.pubtator"
out_pubtator_file="pred_test.pubtator"
in_bert_model="bioredirect_biored_pt"

echo "Converting test pubtator to tsv"
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "${in_pubtator_file}" \
    --out_tsv_file "${in_pubtator_file}.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Running test prediction"
cuda_visible_devices=0 python src/run_exp.py \
    --task_name biored \
    --in_bioredirect_model "${in_bert_model}" \
    --in_test_tsv_file "${in_pubtator_file}.tsv" \
    --out_pred_tsv_file "${in_pubtator_file}.pred.tsv" \
    --batch_size 8

echo "Converting test prediction to pubtator"
python src/run_test_pred.py \
    --to_pubtator3 \
    --in_test_pubtator_file "${in_pubtator_file}" \
    --in_test_tsv_file "${in_pubtator_file}.tsv" \
    --in_pred_tsv_file "${in_pubtator_file}.pred.tsv" \
    --out_pred_pubtator_file "${out_pubtator_file}"

echo "Done"
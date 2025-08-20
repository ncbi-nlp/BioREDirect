#!/bin/bash

format="pubtator"
#format="bioc"
in_data_dir="data"
out_pred_data_dir="outputs"
#sections="ALL"
sections="TITLE|ABSTRACT"

in_bert_model="bioredirect_biored_single_chunk_pt"

echo "Running test prediction"
cuda_visible_devices=0 python src/run_pubtator3_pred.py \
    --format "${format}" \
    --in_bioredirect_model "${in_bert_model}" \
    --in_data_dir "${in_data_dir}" \
    --out_pred_data_dir "${out_pred_data_dir}" \
    --batch_size 8 \
    --sections "${sections}" \
    --use_single_chunk True
echo "Done"
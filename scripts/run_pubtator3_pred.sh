#!/bin/bash

in_bioc_xml_dir="data"
out_pred_bioc_dir="outputs"
#sections="ALL"
sections="TITLE|ABSTRACT"

in_bert_model="bioredirect_biored_pt"

echo "Running test prediction"
cuda_visible_devices=0 python src/run_pubtator3_pred.py \
    --in_bioredirect_model "${in_bert_model}" \
    --in_bioc_xml_dir "${in_bioc_xml_dir}" \
    --out_pred_bioc_dir "${out_pred_bioc_dir}" \
    --batch_size 8

echo "Done"
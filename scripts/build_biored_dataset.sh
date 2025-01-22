#!/bin/bash

in_bert_model="biorex_biolinkbert_pt"

# Convert PubTator to BioREx TSV
echo "Converting train PubTator to TSV..."
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/bioredirect/bioredirect_train.pubtator" \
    --out_tsv_file "datasets/bioredirect/processed/train.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Converting dev PubTator to TSV..."
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/bioredirect/bioredirect_dev.pubtator" \
    --out_tsv_file "datasets/bioredirect/processed/dev.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Converting train_dev PubTator to TSV..."
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/bioredirect/bioredirect_train_dev.pubtator" \
    --out_tsv_file "datasets/bioredirect/processed/train_and_dev.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Converting test PubTator to TSV..."
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/bioredirect/bioredirect_test.pubtator" \
    --out_tsv_file "datasets/bioredirect/processed/test.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Converting bc8 test PubTator to TSV..."
cuda_visible_devices=0 python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/bioredirect/bioredirect_bc8_test.pubtator" \
    --out_tsv_file "datasets/bioredirect/processed/bc8_test.tsv" \
    --in_bert_model "${in_bert_model}"

echo "Done."
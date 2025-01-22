#!/bin/bash

in_bert_model="biorex_biolinkbert_pt"

# Convert PubTator to BioREx TSV
echo "Converting train PubTator to TSV..."
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/cdr/CDR_TrainingSet.PubTator.txt" \
    --out_tsv_file "datasets/cdr/processed/train.tsv" \
    --in_bert_model "${in_bert_model}" \
    --task "cdr"

echo "Converting dev PubTator to TSV..."
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/cdr/CDR_DevelopmentSet.PubTator.txt" \
    --out_tsv_file "datasets/cdr/processed/dev.tsv" \
    --in_bert_model "${in_bert_model}" \
    --task "cdr"
    
echo "Converting test PubTator to TSV..."
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --in_pubtator_file "datasets/cdr/CDR_TestSet.PubTator.txt" \
    --out_tsv_file "datasets/cdr/processed/test.tsv" \
    --in_bert_model "${in_bert_model}" \
    --task "cdr"

cat "datasets/cdr/processed/train.tsv" "datasets/cdr/processed/dev.tsv" > "datasets/cdr/processed/train_dev.tsv"
echo "Done."
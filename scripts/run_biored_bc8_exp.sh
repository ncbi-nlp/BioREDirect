#!/bin/bash

cuda_visible_devices=0 python src/run_exp.py \
    --seed 1111 \
    --task_name biored \
    --in_bert_model biorex_biolinkbert_pt \
    --out_bioredirect_model "out_bioredirect_biored_model" \
    --in_train_tsv_file datasets/bioredirect/processed/train_and_dev.tsv \
    --in_dev_tsv_file datasets/bioredirect/processed/test.tsv \
    --in_test_tsv_file datasets/bioredirect/processed/bc8_test.tsv \
    --soft_prompt_len 8 \
    --num_epochs 10 \
    --batch_size 16 \
    --max_seq_len 512 \
    --learning_rate 1e-5


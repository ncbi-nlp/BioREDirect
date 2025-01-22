#!/bin/bash

cuda_visible_devices=0 python src/run_exp.py \
    --seed 1111 \
    --task_name cdr \
    --in_bert_model biorex_biolinkbert_pt \
    --out_bioredirect_model out_bioredirect_cdr_model \
    --in_train_tsv_file datasets/cdr/processed/train_dev.tsv \
    --in_dev_tsv_file datasets/cdr/processed/train_dev.tsv \
    --in_test_tsv_file datasets/cdr/processed/test.tsv \
    --soft_prompt_len 8 \
    --num_epochs 5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --learning_rate 1e-5
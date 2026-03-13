#!/bin/bash

python eval.py \
  --pred_path data/pred/pred_train_qa_without_cite_meta-llama-3-70b-instruct.jsonl \
  --key_path data/orig/dev/archehr-qa_key.json


#!/bin/bash

python pred.py  \
    --model_name "gpt-5.2-2025-12-11"  \
    --provider_path providers/openai_156.py  \
    --dataset_name "train_qa_without_cite" \
    --api_token OPENAI_TOKEN_GOES_HERE

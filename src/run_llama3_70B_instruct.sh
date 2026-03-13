#!/bin/bash

model_name="meta/meta-llama-3-70b-instruct"
dataset_name="test_qa"

python pred.py  \
    --model_name $model_name  \
    --provider_path providers/replicate_104.py  \
    --dataset_name $dataset_name \
    --api_token REPLICATE_TOKEN_GOES_HERE

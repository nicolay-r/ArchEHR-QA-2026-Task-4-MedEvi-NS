#!/bin/bash

python pred.py  \
    --model_name "meta/meta-llama-3-70b-instruct"  \
    --provider_path providers/replicate_104.py  \
    --dataset_name "test_qa" \
    --api_token REPLICATE_TOKEN_GOES_HERE

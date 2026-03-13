
#!/bin/bash

model_name="gpt-5.2-2025-12-11"
dataset_name="train_qa_without_cite"

python pred.py  \
    --model_name $model_name  \
    --provider_path providers/openai_156.py  \
    --dataset_name $dataset_name \
    --api_token OPENAI_TOKEN_GOES_HERE

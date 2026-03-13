#!/bin/bash
for jsonl_file in ../../data/pred/*test*.jsonl; do
    [ -f "$jsonl_file" ] || continue
    echo "Converting $jsonl_file"
    python3 create.py --pred_jsonl "$jsonl_file" --answers-metadata "../../data/orig/test/archehr-qa_key.json" "../../data/orig/test-2026/archehr-qa_key.json"
done
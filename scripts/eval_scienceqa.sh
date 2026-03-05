#!/bin/bash
# Evaluate ScienceQA results - handles field name mapping (response -> text)
# and question_id type mapping (int -> str)
eval "$(/home/antarachugh/miniconda3/bin/conda shell.bash hook)"
conda activate visattnsink
cd /home/antarachugh/idountang/VisAttnSink

RESULT_FILE="E_answers/llava-v1.5-7b/[ScienceQA-test]scienceqa_7b-1772409399.jsonl"
BASE_DIR="/xuanwu-tank/east/antarachugh/projects/VisAttnSink_data/ScienceQA"

# Convert response -> text and int question_id -> str for eval compatibility
python3 -c "
import json, sys

# Read and convert
lines = open('$RESULT_FILE').readlines()
converted = []
for line in lines:
    d = json.loads(line)
    # Strip </s> from response and rename to text
    d['text'] = d.pop('response', '').replace('</s>', '').strip()
    # Ensure question_id is string (for problems.json lookup)
    d['question_id'] = str(d['question_id'])
    converted.append(d)

# Write converted file
out = 'E_answers/llava-v1.5-7b/scienceqa_7b_eval_ready.jsonl'
with open(out, 'w') as f:
    for d in converted:
        f.write(json.dumps(d) + '\n')
print(f'Converted {len(converted)} lines -> {out}')
"

# Run eval
python src/eval/eval_science_qa.py \
    --base-dir "$BASE_DIR" \
    --result-file "E_answers/llava-v1.5-7b/scienceqa_7b_eval_ready.jsonl" \
    --output-file "E_answers/llava-v1.5-7b/eval_details.json" \
    --output-result "E_answers/llava-v1.5-7b/eval_results.json"

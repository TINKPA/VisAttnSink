#!/bin/bash
eval "$(/home/antarachugh/miniconda3/bin/conda shell.bash hook)"
conda activate visattnsink
cd /home/antarachugh/idountang/VisAttnSink
CUDA_VISIBLE_DEVICES=0 python src/inference.py --exp_config A_exps/scienceqa_7b.yml --device 0

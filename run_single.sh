#!/bin/bash
eval "$(/home/antarachugh/miniconda3/bin/conda shell.bash hook)"
conda activate visattnsink
cd /home/antarachugh/idountang/VisAttnSink
CUDA_VISIBLE_DEVICES=1 python run_single_sample.py

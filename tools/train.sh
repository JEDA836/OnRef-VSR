#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=7271 tools/train.py \
                       configs/OnRef-VSR_reds4_cubic.py \
                        --seed 0 \


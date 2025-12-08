py3clean ./
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=7049 tools/test.py \
                       configs.py \
                        model_path \
                        --seed 0 \
                        --save-path result \

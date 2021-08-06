#!/bin/bash  
python3.7 -m paddle.distributed.launch \
          --gpus='0,1,2,3' \
          --log_dir='log' \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.80 \
          --lr 0.005 \
          --num_epochs 105 \
          --test_period 5 \
          --model_path "./model-distill-80" \
          --initial_ratio 0.15 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --step_epochs 71 88


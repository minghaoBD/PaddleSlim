#!/bin/bash  
python3.7 train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --model MobileNet \
          --num_epochs 108 \
          --test_period 5 \
          --model_path "./models" \
          --initial_ratio 0.15 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --step_epochs 71 88

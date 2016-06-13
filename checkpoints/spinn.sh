#!/usr/bin/env bash

source _init.sh

# The invocation below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs, and delete the flag --ckpt_path ... to train from scratch.

# SPINN
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn.ckpt_best --connect_tracking_comp  --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.938514416034 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.76018187539e-05 --learning_rate 0.00103428201391 --log_path $LOG_DIR  --model_dim 600 --model_type Model1 --num_sentence_pair_combination_layers 1 --semantic_classifier_keep_rate 0.949455648614 --seq_length 50 --tracking_lstm_hidden_dim 44 --training_data_path $SNLI_TRAIN_JSONL --transition_cost_scale 0.605159568546 --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn "

python -m spinn.models.fat_classifier $REMBED_FLAGS


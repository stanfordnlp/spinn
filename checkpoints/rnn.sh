#!/usr/bin/env bash

source _init.sh

# The invocation below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs, and delete the flag --ckpt_path ... to train from scratch.

# RNN
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/rnn.ckpt_best --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.852564448733 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 25 --init_range 0.005 --l2_lambda 4.42556134893e-06 --learning_rate 0.00464868093302 --log_path $LOG_DIR  --model_dim 600 --model_type RNN --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.883392584372 --seq_length 25 --tracking_lstm_hidden_dim 33 --training_data_path $SNLI_TRAIN_JSONL --word_embedding_dim 300 --experiment_name rnn "

python -m spinn.models.fat_classifier $REMBED_FLAGS


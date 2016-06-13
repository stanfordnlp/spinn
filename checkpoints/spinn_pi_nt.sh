#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

source _init.sh

# The invocation below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs, and delete the flag --ckpt_path ... to train from scratch.

# SPINN-PI-NT
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn_pi_nt.ckpt_best --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.828528124124 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 50 --init_range 0.005 --l2_lambda 3.45058959758e-06 --learning_rate 0.000297682444894 --log_path $LOG_DIR  --model_dim 600 --model_type Model0 --noconnect_tracking_comp  --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.9437038157 --seq_length 50 --tracking_lstm_hidden_dim 57 --training_data_path $SNLI_TRAIN_JSONL --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn_pi_nt "

python -m spinn.models.fat_classifier $REMBED_FLAGS


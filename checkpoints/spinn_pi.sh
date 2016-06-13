#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

export MODEL="spinn_pi"
source _init.sh

# The invocation below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs, and delete the flag --ckpt_path ... to train from scratch.

# SPINN-PI
export REMBED_FLAGS="  --batch_size 32 --connect_tracking_comp --embedding_keep_rate 0.917969380132 --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.00098223698e-05 --learning_rate 0.00701855401337 --model_dim 600 --model_type Model0 --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.934741728838 --seq_length 50 --tracking_lstm_hidden_dim 61 --use_tracking_lstm  --word_embedding_dim 300"

echo "Running command:"
echo "python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS"
python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS



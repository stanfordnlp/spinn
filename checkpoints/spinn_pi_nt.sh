#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

export MODEL=spinn_pi_nt
source _init.sh

# SPINN-PI-NT
export REMBED_FLAGS="  --batch_size 32 --embedding_keep_rate 0.828528124124 --eval_seq_length 50 --init_range 0.005 --l2_lambda 3.45058959758e-06 --learning_rate 0.000297682444894 --model_dim 600 --model_type Model0 --noconnect_tracking_comp  --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.9437038157 --seq_length 50 --tracking_lstm_hidden_dim 57 --use_tracking_lstm  --word_embedding_dim 300 "

echo "Running command:"
echo "python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS"
python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS


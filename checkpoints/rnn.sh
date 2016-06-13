#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

export MODEL="rnn"
source _init.sh

# RNN
export REMBED_FLAGS="  --batch_size 32 --embedding_keep_rate 0.852564448733 --eval_seq_length 25 --init_range 0.005 --l2_lambda 4.42556134893e-06 --learning_rate 0.00464868093302 --model_dim 600 --model_type RNN --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.883392584372 --seq_length 25 --tracking_lstm_hidden_dim 33 --word_embedding_dim 300"

echo "Running command:"
echo "python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS"
python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS


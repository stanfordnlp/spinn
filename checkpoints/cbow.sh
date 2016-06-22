#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

export MODEL="cbow"
source _init.sh

# RNN
export REMBED_FLAGS="  --batch_size 32 --eval_seq_length 25 --init_range 0.005 --l2_lambda 1.24280631663e-07 --learning_rate 0.00829688998827 --model_dim 300 --model_type CBOW --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.88010692672 --seq_length 25 --tracking_lstm_hidden_dim 33 --word_embedding_dim 300"

echo "Running command:"
echo "python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS"
python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS


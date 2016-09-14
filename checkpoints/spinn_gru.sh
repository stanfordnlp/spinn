#!/usr/bin/env bash

if [ -e README.md ]; then
    cd checkpoints
fi

export MODEL="spinn_gru"
source _init.sh

# SPINN
export REMBED_FLAGS="  --use_gru --batch_size 32 --connect_tracking_comp --embedding_keep_rate 0.938514416034 --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.76018187539e-05 --learning_rate 0.00103428201391 --model_dim 600 --model_type Model1 --num_sentence_pair_combination_layers 1 --semantic_classifier_keep_rate 0.949455648614 --seq_length 50 --tracking_lstm_hidden_dim 0 --transition_cost_scale 0.605159568546 --nouse_tracking_lstm  --word_embedding_dim 300 --nopredict_use_cell --noconnect_tracking_comp"

echo "Running command:"
echo "python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS"
python -m spinn.models.fat_classifier $BASE_FLAGS $REMBED_FLAGS


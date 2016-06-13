#!/usr/bin/env bash

# Script to do snli runs from pretrained models.

# The directory where logs should be stored.
export LOG_DIR=logs
mkdir -p $LOG_DIR

# The path to pretrained embeddings.
export EMBEDDING_PATH=../glove/glove.840B.300d.txt
if [ ! -e "$EMBEDDING_PATH" ]; then
    echo "Could not find GloVe embeddings at $EMBEDDING_PATH."
    read -p "Should we download them? (~2G download) (y/n) " yn
    if echo "$yn" | grep -iq "[^yY]"; then
        exit 1
    fi

    mkdir -p `dirname "$EMBEDDING_PATH"`
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.zip \
        || (echo "Failed to download GloVe embeddings." >&2 && exit 1)
    unzip -d `dirname "$EMBEDDING_PATH"` glove.zip && rm glove.zip
fi

# Prepare SNLI data.
export SNLI_DIR=../snli_1.0
export SNLI_TRAIN_JSONL=$SNLI_DIR/snli_1.0_train.jsonl
export SNLI_DEV_JSONL=$SNLI_DIR/snli_1.0_dev.jsonl
export SNLI_TEST_JSONL=$SNLI_DIR/snli_1.0_test.jsonl
if [ ! -d "$SNLI_DIR" ]; then
    echo "Downloading SNLI data." >&2
    wget http://nlp.stanford.edu/projects/snli/snli_1.0.zip -O snli_1.0.zip \
        || (echo "Failed to download SNLI data." >&2 && exit 1)
    unzip -d .. snli_1.0.zip && rm snli_1.0.zip
fi

export PYTHONPATH=../python
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32

# The invocations below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs, and delete the flag --ckpt_path ... to train from scratch.

# RNN
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/rnn.ckpt_best --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.852564448733 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 25 --init_range 0.005 --l2_lambda 4.42556134893e-06 --learning_rate 0.00464868093302 --log_path $LOG_DIR  --model_dim 600 --model_type RNN --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.883392584372 --seq_length 25 --tracking_lstm_hidden_dim 33 --training_data_path $SNLI_TRAIN_JSONL --word_embedding_dim 300 --experiment_name rnn "

python -m spinn.models.fat_classifier $REMBED_FLAGS

# SPINN-PI-NT
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn_pi_nt.ckpt_best --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.828528124124 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 50 --init_range 0.005 --l2_lambda 3.45058959758e-06 --learning_rate 0.000297682444894 --log_path $LOG_DIR  --model_dim 600 --model_type Model0 --noconnect_tracking_comp  --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.9437038157 --seq_length 50 --tracking_lstm_hidden_dim 57 --training_data_path $SNLI_TRAIN_JSONL --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn_pi_nt "

python -m spinn.models.fat_classifier $REMBED_FLAGS

# SPINN-PI
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn_pi.ckpt_best --connect_tracking_comp  --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.917969380132 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.00098223698e-05 --learning_rate 0.00701855401337 --log_path $LOG_DIR  --model_dim 600 --model_type Model0 --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.934741728838 --seq_length 50 --tracking_lstm_hidden_dim 61 --training_data_path $SNLI_TRAIN_JSONL --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn_pi "

python -m spinn.models.fat_classifier $REMBED_FLAGS

# SPINN
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn.ckpt_best --connect_tracking_comp  --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.938514416034 --eval_data_path $SNLI_DEV_JSONL --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.76018187539e-05 --learning_rate 0.00103428201391 --log_path $LOG_DIR  --model_dim 600 --model_type Model1 --num_sentence_pair_combination_layers 1 --semantic_classifier_keep_rate 0.949455648614 --seq_length 50 --tracking_lstm_hidden_dim 44 --training_data_path $SNLI_TRAIN_JSONL --transition_cost_scale 0.605159568546 --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn "

python -m spinn.models.fat_classifier $REMBED_FLAGS


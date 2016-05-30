#!/usr/bin/env bash

# Script to do snli runs from pretrained models.

# The directory where logs should be stored. Must exist.
export LOG_DIR=logs
# The path to pretrained embeddings.
export EMBEDDING_PATH=../glove/glove.840B.300d.txt

export PYTHONPATH=../python

# The invocations below will load the pretrained models and continue training
# by default. Use the flag --expanded_eval_only mode to do eval-only runs.

# SPINN-PI-NT
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn_pi_nt.ckpt_best --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.828528124124 --eval_data_path snli_1.0/snli_1.0_dev.jsonl --eval_seq_length 50 --init_range 0.005 --l2_lambda 3.45058959758e-06 --learning_rate 0.000297682444894 --log_path $LOG_DIR --lstm_composition  --model_dim 600 --model_type Model0 --noconnect_tracking_comp  --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.9437038157 --seq_length 50 --tracking_lstm_hidden_dim 57 --training_data_path snli_1.0/snli_1.0_train.jsonl --transition_cost_scale 0.90840499834 --use_attention None --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn_pi_nt "

THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32 python -m spinn.models.fat_classifier $REMBED_FLAGS

# SPINN-PI
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn_pi.ckpt_best --connect_tracking_comp  --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.917969380132 --eval_data_path snli_1.0/snli_1.0_dev.jsonl --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.00098223698e-05 --learning_rate 0.00701855401337 --log_path $LOG_DIR --lstm_composition  --model_dim 600 --model_type Model0 --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.934741728838 --seq_length 50 --tracking_lstm_hidden_dim 61 --training_data_path snli_1.0/snli_1.0_train.jsonl --transition_cost_scale 0.590827336331 --use_attention None --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn_pi "

THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32 python -m spinn.models.fat_classifier $REMBED_FLAGS

# SPINN
export REMBED_FLAGS="  --batch_size 32 --ckpt_path checkpoints/spinn.ckpt_best --connect_tracking_comp  --data_type snli --embedding_data_path $EMBEDDING_PATH --embedding_keep_rate 0.938514416034 --eval_data_path snli_1.0/snli_1.0_dev.jsonl --eval_seq_length 50 --init_range 0.005 --l2_lambda 2.76018187539e-05 --learning_rate 0.00103428201391 --log_path $LOG_DIR --lstm_composition  --model_dim 600 --model_type Model1 --num_sentence_pair_combination_layers 1 --semantic_classifier_keep_rate 0.949455648614 --seq_length 50 --tracking_lstm_hidden_dim 44 --training_data_path snli_1.0/snli_1.0_train.jsonl --transition_cost_scale 0.605159568546 --use_attention None --use_tracking_lstm  --word_embedding_dim 300 --experiment_name spinn "

THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32 python -m spinn.models.fat_classifier $REMBED_FLAGS


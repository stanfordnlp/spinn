#!/usr/bin/env bash

# Header script to prepare to perform a SPINN SNLI run.

function show_help {
    echo "$(basename "$0") [-h] [-e] [-t] [-s] -- run a train or test run of a SPINN model

    where:
        -h    show this help text
        -e    run in eval-only mode (evaluates on dev set by default)
        -t    evaluate on test set
        -s    skip the checkpoint loading; run with a randomly initialized model"
    exit
}

# Parse arguments
eval_only=false
test_set=false
skip_ckpt=false
while [[ $# > 0 ]]; do
    case "$1" in
        -e)
        eval_only=true
        ;;
        -t)
        test_set=true
        ;;
        -s)
        skip_ckpt=true
        ;;
        -h|--help)
        show_help
        ;;
        *)
        ;;
    esac
    shift
done

# The directory where logs should be stored.
export LOG_DIR=../logs
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

flags="--data_type snli --embedding_data_path $EMBEDDING_PATH --log_path $LOG_DIR --training_data_path $SNLI_TRAIN_JSONL --experiment_name $MODEL"
if [ "$eval_only" = true ]; then
    flags="$flags --expanded_eval_only"
fi
if [ "$test_set" = true ]; then
    flags="$flags --eval_data_path $SNLI_TEST_JSONL"
else
    flags="$flags --eval_data_path $SNLI_DEV_JSONL"
fi
if [ ! "$skip_ckpt" = "true" ]; then
    flags="$flags --ckpt_path ${MODEL}.ckpt_best"
fi
export BASE_FLAGS="$flags"

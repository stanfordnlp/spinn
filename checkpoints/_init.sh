#!/usr/bin/env bash

# Header script to prepare to perform a SPINN SNLI run.

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

#ifndef _sequence_model_
#define _sequence_model_

#include <cuda_runtime.h>

#include "blocks.h"
#include "util.h"

#include "kernels.cuh"


struct SequenceModelSpec {
  size_t model_dim;
  size_t word_embedding_dim;
  size_t batch_size;
  size_t vocab_size;
  size_t seq_length;
  size_t model_visible_dim;

  SequenceModelSpec(size_t model_dim, size_t word_embedding_dim,
                    size_t batch_size, size_t vocab_size, size_t seq_length,
                    size_t model_visible_dim)
    : model_dim(model_dim), word_embedding_dim(word_embedding_dim),
      batch_size(batch_size), vocab_size(vocab_size), seq_length(seq_length),
      model_visible_dim(model_visible_dim) {};

};


class SequenceModel {

  public:

    SequenceModelSpec spec;
    SequenceModel(SequenceModelSpec spec)
      : spec(spec), X_indices(spec.batch_size * spec.seq_length),
        X(spec.batch_size * spec.model_dim, spec.seq_length) {};

    // Embedding index inputs, of dimension `batch_size * seq_length` -- i.e.,
    // we have `seq_length`-many concatenated vectors of embedding integers
    vec X_indices;
    // Embedding inputs, of dimension `model_dim * (batch_size * seq_length)` --
    // i.e., along 2nd axis we have `seq_length`-many `model_dim * batch_size`
    // matrices.
    mat X;

    void lookup_embeddings(mat& embedding_source) {
      embedding_source.subtensor1(X, X_indices, 0.0f, 1.0f, 0.0f, NULL);
    }

    virtual void forward() = 0;

};


#endif

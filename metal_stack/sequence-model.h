#ifndef _sequence_model_
#define _sequence_model_

#include <cuda_runtime.h>

#include "util.h"

#include "kernels.cuh"

class SequenceModel {

  public:

    ModelSpec spec;
    SequenceModel(ModelSpec spec) : spec(spec) {};

    // Embedding index inputs, of dimension `batch_size * seq_length` -- i.e.,
    // we have `seq_length`-many concatenated vectors of embedding integers
    float *X_indices;
    // Embedding inputs, of dimension `model_dim * (batch_size * seq_length)` --
    // i.e., along 2nd axis we have `seq_length`-many `model_dim * batch_size`
    // matrices.
    float *X;

    void lookup_embeddings(float *embedding_source) {
      kernels::subtensor1(X, embedding_source, X_indices, spec.vocab_size,
          spec.seq_length * spec.batch_size, spec.model_dim, 0.0f, 1.0f, 0.0f,
          NULL);
    }

    virtual void forward() = 0;

};


#endif

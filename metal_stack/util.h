#ifndef _util_
#define _util_

#include <cstddef>
#include <iostream>

typedef struct ModelSpec {
  size_t model_dim;
  size_t word_embedding_dim;
  size_t batch_size;
  size_t vocab_size;
  size_t seq_length;
  size_t model_visible_dim;
} ModelSpec;

#endif

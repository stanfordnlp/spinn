#ifndef _util_
#define _util_

#include <cstddef>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <curand.h>

using namespace std;


typedef struct ModelSpec {
  size_t model_dim;
  size_t word_embedding_dim;
  size_t batch_size;
  size_t vocab_size;
  size_t seq_length;
  size_t model_visible_dim;
} ModelSpec;


float *load_weights(string filename, int N);
float *load_weights_cuda(string filename, int N);

void print_device_matrix(float *m, int M, int N);
void fill_rand_matrix(float *m, int M, int N);

#endif

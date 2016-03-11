#ifndef _util_
#define _util_

#include <cstddef>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

using namespace std;


typedef struct ModelSpec {
  size_t model_dim;
  size_t word_embedding_dim;
  size_t batch_size;
  size_t vocab_size;
  size_t seq_length;
  size_t model_visible_dim;
} ModelSpec;


float* load_weights(string filename, int N) {
  float *ret = (float *) malloc(N * sizeof(float));
  ifstream file(filename);

  float x;
  for (int i = 0; i < N; i++) {
    file >> x;
    cout << x << endl;
    ret[i] = x;
  }
}

float *load_weights_cuda(string filename, int N) {
  float *h_weights = load_weights(filename, N);
  float *d_weights;
  cudaMalloc(&d_weights, N * sizeof(float));
  cudaMemcpy(d_weights, h_weights, N * sizeof(float),
      cudaMemcpyHostToDevice);
  return d_weights;
}

#endif

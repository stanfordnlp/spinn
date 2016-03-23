#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "rnn.h"
#include "util.h"


RNNParameters load_params(ModelSpec spec) {
  float *U = make_rand_matrix(spec.model_dim, spec.model_dim);
  float *W = make_rand_matrix(spec.model_dim, spec.word_embedding_dim);
  float *b = make_rand_matrix(spec.model_dim, 1);

  RNNParameters ret = {U, W, b};
  return ret;
}

void destroy_params(RNNParameters params) {
  cudaFree(params.U);
  cudaFree(params.W);
  cudaFree(params.b);
}

int main() {
  ModelSpec spec = {300, 300, (size_t) atoi(getenv("BATCH_SIZE")), 10, 30, 300};
  RNNParameters params = load_params(spec);

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "CUBLAS initialization failed" << endl;
    return 1;
  }

  RNN rnn(spec, params, handle);

  // Set model inputs.
  cout << "X:" << endl;
  fill_rand_matrix(rnn.X, spec.model_dim, spec.batch_size * spec.seq_length);

  auto time_elapsed = chrono::microseconds::zero();
  int n_batches = 50;
  for (int t = 0; t < n_batches; t++) {
    auto start = chrono::high_resolution_clock::now();
    rnn.forward();
    auto end = chrono::high_resolution_clock::now();
    time_elapsed += chrono::duration_cast<chrono::microseconds>(end - start);
  }

  // Print the final representation.
  cout << "Output:" << endl;
  print_device_matrix(rnn.output, spec.model_dim, spec.batch_size);

  cout << "Total time elapsed: " << time_elapsed.count() << endl;

  destroy_params(params);
}

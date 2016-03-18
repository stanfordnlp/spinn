#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "thin-stack.h"
#include "util.h"


ThinStackParameters load_params(ModelSpec spec) {
  float *compose_W_l = load_weights_cuda("params/compose_W_l.txt",
      spec.model_dim * spec.model_dim);
  float *compose_W_r = load_weights_cuda("params/compose_W_r.txt",
      spec.model_dim * spec.model_dim);
  float *compose_b = load_weights_cuda("params/compose_b.txt", spec.model_dim);

  ThinStackParameters ret = {
    NULL, NULL, NULL, // tracking
    compose_W_l, compose_W_r, NULL, compose_b, // composition
  };

  return ret;
}

void destroy_params(ThinStackParameters params) {
  cudaFree(params.compose_W_l);
  cudaFree(params.compose_W_r);
  cudaFree(params.compose_b);
}

int main() {
  ModelSpec spec = {300, 300, (size_t) atoi(getenv("BATCH_SIZE")), 10, 59, 300};
  ThinStackParameters params = load_params(spec);

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "CUBLAS initialization failed" << endl;
    return 1;
  }

  ThinStack ts(spec, params, handle);

  // Set model inputs.
  cout << "X:" << endl;
  int num_tokens = (spec.seq_length + 1) / 2;
  fill_rand_matrix(ts.X, spec.model_dim, spec.batch_size * num_tokens);

  cout << "transitions:" << endl;
  float *h_transitions = (float *) malloc(spec.seq_length * spec.batch_size * sizeof(float));
  for (int i = 0; i < spec.seq_length * spec.batch_size; i++) {
    float val;
    if (i < spec.batch_size * 2) {
      val = 0.0f;
    } else if (i >= spec.batch_size * 2 && i < spec.batch_size * 3) {
      val = 1.0f;
    } else {
      val = rand() % 2 == 0 ? 1.0f : 0.0f;
    }
    h_transitions[i] = val;
  }
  cudaMemcpy(ts.transitions, h_transitions, spec.seq_length * spec.batch_size * sizeof(float), cudaMemcpyHostToDevice);
  free(h_transitions);
#if DEBUG
  print_device_matrix(ts.transitions, 1, spec.batch_size * spec.seq_length);
#endif

  auto time_elapsed = chrono::microseconds::zero();
  int n_batches = 50;
  for (int t = 0; t < n_batches; t++) {
    auto start = chrono::high_resolution_clock::now();
    ts.forward();
    auto end = chrono::high_resolution_clock::now();
    time_elapsed += chrono::duration_cast<chrono::microseconds>(end - start);
  }

  // Print the top of the stack.
  cout << "Stack top:" << endl;
  print_device_matrix(&ts.stack[(spec.seq_length - 1) * spec.model_dim * spec.batch_size], spec.model_dim, spec.batch_size);

  cout << "Total time elapsed: " << time_elapsed.count() << endl;

  destroy_params(params);
}

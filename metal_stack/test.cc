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

  ThinStackParameters ret = {
    NULL, NULL, // projection
    NULL, NULL, // buffer batch-norm
    NULL, NULL, NULL, // tracking
    compose_W_l, compose_W_r, NULL, NULL // composition
  };

  return ret;
}

void destroy_params(ThinStackParameters params) {
  cudaFree(params.compose_W_l);
  cudaFree(params.compose_W_r);
}

int main() {
  ModelSpec spec = {5, 5, 2, 10, 3, 5};//{300, 300, 256, 10, 25, 300};//{5, 5, 2, 10, 3, 5};
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
  fill_rand_matrix(ts.X, spec.model_dim, spec.batch_size * spec.seq_length);
#if DEBUG
  print_device_matrix(ts.X, spec.model_dim, spec.batch_size * spec.seq_length);
#endif

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
  //cudaMemset(ts.transitions, 0, spec.batch_size * spec.seq_length * sizeof(int));
  cudaMemcpy(ts.transitions, h_transitions, spec.seq_length * spec.batch_size * sizeof(float), cudaMemcpyHostToDevice);
  free(h_transitions);
#if DEBUG
  print_device_matrix(ts.transitions, spec.batch_size, spec.seq_length);
#endif

  auto time_elapsed = chrono::microseconds::zero();
  int n_batches = 1; // 50
  for (int t = 0; t < n_batches; t++) {
    auto start = chrono::high_resolution_clock::now();
    ts.forward();
    auto end = chrono::high_resolution_clock::now();
    time_elapsed += chrono::duration_cast<chrono::microseconds>(end - start);
  }

  cout << "Total time elapsed: " << time_elapsed.count() << endl;

  destroy_params(params);
}

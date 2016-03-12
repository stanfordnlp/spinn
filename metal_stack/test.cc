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
  ModelSpec spec = {5, 5, 2, 10, 3, 5};
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
  print_device_matrix(ts.X, spec.model_dim, spec.batch_size * spec.seq_length);
  cout << "transitions:" << endl;
  cudaMemset(ts.transitions, 0, spec.batch_size * spec.seq_length * sizeof(int));

  ts.forward();

  destroy_params(params);
}

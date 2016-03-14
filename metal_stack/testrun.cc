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
  ModelSpec spec = {50, 50, 1, 10, 71, 50};//{300, 300, 256, 10, 25, 300};//{5, 5, 2, 10, 3, 5};
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
  load_weights_cuda("/scr/jgauthie/projects/rembed/metal_stack/sst_one_buffer.txt", spec.model_dim * spec.batch_size * num_tokens, ts.X);
/* #if DEBUG */
/*   print_device_matrix(ts.X, spec.model_dim, spec.batch_size * num_tokens); */
/* #endif */

  cout << "transitions:" << endl;
  load_weights_cuda("/scr/jgauthie/tmp/deep-recursive/models/drsv_1_50_0_0.1_1_200_0.002_0.0001_0.9_0.transitions", spec.batch_size * spec.seq_length, ts.transitions);
#if DEBUG
  print_device_matrix(ts.transitions, 1, spec.batch_size * spec.seq_length);
#endif

  auto time_elapsed = chrono::microseconds::zero();
  int n_batches = 1; // 50
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

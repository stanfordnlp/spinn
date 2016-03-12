#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "thin-stack.h"
#include "util.h"

using namespace testing;

static ThinStack make_stack(ModelSpec spec) {
  // Make up random parameters.
  float *compose_W_l, *compose_W_r;
  cudaMalloc(&compose_W_l, spec.model_dim * spec.model_dim * sizeof(float));
  cudaMalloc(&compose_W_r, spec.model_dim * spec.model_dim * sizeof(float));
  fill_rand_matrix(compose_W_l, spec.model_dim, spec.model_dim);
  fill_rand_matrix(compose_W_r, spec.model_dim, spec.model_dim);

  ThinStackParameters params = {
    NULL, NULL, // projection
    NULL, NULL, // buffer batch-norm
    NULL, NULL, NULL, // tracking
    compose_W_l, compose_W_r, NULL, NULL // composition
  };
  cublasHandle_t handle = getCublasHandle();
  ThinStack ts(spec, params, handle);
  return ts;
}

static void free_stack(ThinStack s) {
  cudaFree(s.params.compose_W_l);
  cudaFree(s.params.compose_W_r);
  cublasDestroy(s.handle);
}

static inline void assert_matrices_equal(const float *m1, const float *m2,
    int M, int N) {
  cudaDeviceSynchronize();
  float *h_m1 = (float *) malloc(2 * M * N * sizeof(float));
  float *h_m2 = &h_m1[M * N];

  cudaMemcpy(h_m1, m1, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_m2, m2, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ASSERT_THAT(h_m1[j * M + i], FloatEq(h_m1[j * M + i]));
    }
  }

  free(h_m1);
}


// Test simple shift-shift-merge feedforward with live random weights.
TEST(ThinStackTest, ShiftShiftMerge) {

  // TODO: Try with larger batch size, model dim
  ModelSpec spec = {5, 5, 2, 10, 3, 5};
  ThinStack ts = make_stack(spec);

  // Make up random inputs.
  fill_rand_matrix(ts.X, spec.model_dim, spec.seq_length * spec.batch_size);

  float h_transitions[] = {
    0.0f, 0.0f,
    0.0f, 0.0f,
    1.0f, 1.0f
  };
  cublasSetVector(spec.seq_length * spec.batch_size, sizeof(float),
      h_transitions, 1, ts.transitions, 1);

  // Do the feedforward!
  ts.forward();

  // Now simulate the feedforward -- this should just be the composition of the
  // first two buffer elements.
  float *expected;
  cudaMalloc(&expected, spec.model_dim * spec.batch_size * sizeof(float));

  // W_l l
  float alpha = 1.0f, beta = 0.0f;
  float *left_child = &ts.X[0];
  cublasSgemm(ts.handle, CUBLAS_OP_N, CUBLAS_OP_N, spec.model_dim, spec.batch_size,
      spec.model_dim, &alpha, ts.params.compose_W_l, spec.model_dim,
      left_child, spec.model_dim, &beta, expected, spec.model_dim);

  // += W_r r
  float beta2 = 1.0f;
  float *right_child = &ts.X[spec.model_dim * spec.batch_size];
  cublasSgemm(ts.handle, CUBLAS_OP_N, CUBLAS_OP_N, spec.model_dim, spec.batch_size,
      spec.model_dim, &alpha, ts.params.compose_W_r, spec.model_dim,
      right_child, spec.model_dim, &beta2, expected, spec.model_dim);

  float *output = &ts.stack[0];
  assert_matrices_equal(output, expected, spec.model_dim, spec.batch_size);

  free_stack(ts);

}

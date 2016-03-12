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

  cout << "compose_W_l" << endl;
  print_device_matrix(compose_W_l, spec.model_dim, spec.model_dim);

  cout << "compose_W_r" << endl;
  print_device_matrix(compose_W_r, spec.model_dim, spec.model_dim);

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
      ASSERT_THAT(h_m1[j * M + i], FloatEq(h_m2[j * M + i]));
    }
  }

  free(h_m1);
}

static float *compose(float *dst, ThinStack& ts, const float *l,
    const float *r) {
  // W_l l
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(ts.handle, CUBLAS_OP_N, CUBLAS_OP_N, ts.spec.model_dim,
      ts.spec.batch_size, ts.spec.model_dim, &alpha, ts.params.compose_W_l,
      ts.spec.model_dim, l, ts.spec.model_dim, &beta, dst, ts.spec.model_dim);

  // += W_r r
  float beta2 = 1.0f;
  cublasSgemm(ts.handle, CUBLAS_OP_N, CUBLAS_OP_N, ts.spec.model_dim,
      ts.spec.batch_size, ts.spec.model_dim, &alpha, ts.params.compose_W_r,
      ts.spec.model_dim, r, ts.spec.model_dim, &beta2, dst, ts.spec.model_dim);

  return dst;
}


class ThinStackTest : public ::testing::Test {

  public:

    ModelSpec spec;
    ThinStack ts;

    ThinStackTest() :
      spec({5, 5, 2, 10, 5, 5}),
      ts(make_stack(spec)) {

      fill_rand_matrix(ts.X, spec.model_dim, spec.seq_length * spec.batch_size);

    }

    virtual void TearDown() {
      free_stack(ts);
    }

};


// Test simple shift-shift-merge feedforward with live random weights.
TEST_F(ThinStackTest, ShiftShiftMerge) {

  float h_transitions[] = {
    0.0f, 0.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 0.0f, // DUMMY
    0.0f, 0.0f, // DUMMY
  };
  cublasSetVector(spec.seq_length * spec.batch_size, sizeof(float),
      h_transitions, 1, ts.transitions, 1);

  // Do the feedforward!
  ts.forward();

  // Now simulate the feedforward -- this should just be the composition of the
  // first two buffer elements.
  float *expected;
  cudaMalloc(&expected, spec.model_dim * spec.batch_size * sizeof(float));

  float *left_child = &ts.X[0];
  float *right_child = &ts.X[spec.model_dim * spec.batch_size];
  compose(expected, ts, left_child, right_child);

  float *output = &ts.stack[2 * spec.model_dim * spec.batch_size];
  assert_matrices_equal(output, expected, spec.model_dim, spec.batch_size);

}


TEST_F(ThinStackTest, ShiftShiftMergeShiftMerge) {

  float h_transitions[] = {
    0.0f, 0.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 0.0f,
    1.0f, 1.0f,
  };
  cublasSetVector(spec.seq_length * spec.batch_size, sizeof(float),
      h_transitions, 1, ts.transitions, 1);

  // Do the feedforward!
  ts.forward();

  // Now simulate the feedforward.
  float *c1, *c2;
  cudaMalloc(&c1, spec.model_dim * spec.batch_size * sizeof(float));
  cudaMalloc(&c2, spec.model_dim * spec.batch_size * sizeof(float));

  // c1
  float *left_child = &ts.X[0];
  float *right_child = &ts.X[spec.model_dim * spec.batch_size];
  compose(c1, ts, left_child, right_child);

  // c2
  left_child = c1;
  right_child = &ts.X[2 * spec.model_dim * spec.batch_size];
  compose(c2, ts, left_child, right_child);

  float *output = &ts.stack[4 * spec.model_dim * spec.batch_size];
  assert_matrices_equal(output, c2, spec.model_dim, spec.batch_size);

}

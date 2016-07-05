#include "blocks.h"
namespace k = kernels;

#include <cuda_runtime.h>
#include "cublas_v2.h"


void batch_normed_mlp(int m, int n, BatchNormedMLPParameters params,
                      const float *inp, float *out, cublasHandle_t handle) {

  xw_plus_b(m, n, params.W, params.b, inp, out, handle);

  // relu
  k::relu(out, m, n);

  // batch-norm
  // - mean
  k::addi_mv(out, params.bn_tm, -1.0, m, n);
  // * gamma
  k::muli_mv(out, params.bn_g, 1.0, m, n);
  // / std
  k::divi_mv(out, params.bn_ts, 1.0, m, n);
  // + beta
  k::addi_mv(out, params.bn_b, 1.0, m, n);

}


void xw_plus_b(int m, int n, const float *W, const float *b, const float *x,
               float *out, cublasHandle_t handle) {

  // out = W x
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &alpha, W,
      n, inp, n, &beta, out, n);

  // out += b
  k::addi_mv(out, b, 1.0, m, n);

}


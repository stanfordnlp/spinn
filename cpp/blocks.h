#ifndef _blocks_
#define _blocks_

#include "kernels.cuh"


typedef struct BatchNormedMLPParameters {
  float *W;
  float *b;

  float *bn_g;
  float *bn_b;
  float *bn_ts;
  float *bn_tm;
} BatchNormedMLPParameters;

void batch_normed_mlp(int m, int n, BatchNormedMLPParameters params,
                      const float *inp, float *out, cublasHandle_t handle);


void xw_plus_b(int m, int n, const float *W, const float *b, const float *x,
               float *out, cublasHandle_t handle);



#endif

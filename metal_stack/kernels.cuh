#ifndef _kernels_
#define _kernels_

#include <cstdlib>

#include <cuda_runtime.h>
#include "cublas_v2.h"


// Grid dimension constraints for jagupard machines.
#define MAX_BLOCKS 65535
#define MAX_THREADS_PER_BLOCK 1024


namespace kernels {

  // v *= s
  void muli_vs(float *v, float s, int N);
  __global__ void k_muli_vs(float *v, float s, int N);

  /**
   * Add two vectors inplace (writing to the first).
   *
   *     v1 = v1 * v1_coeff + v2 * v2_coeff
   */
  void addi_vv(cublasHandle_t handle, float *v1, const float *v2,
          float v1_coeff, float v2_coeff, int N);

  /**
   * Broadcast-add an `N`-dim column vector onto an `M * N` matrix.
   *
   *     m += coeff * v
   */
  void addi_mv(float *m, const float *v, float coeff, int M, int N);
  __global__ void k_addi_mv(float *m, const float *v, float coeff, int M,
      int N);

  void relu(float *m, int M, int N);
  __global__ void k_relu(float *m, int M, int N);

  /**
   * Retrieve a subset of `N` rows from the contiguous `src_N * D` matrix `src`
   * and write them to `dst` (`M >= N`). `dst` should be large enough to hold
   * the `N * D` float result. `idxs` should be a length-`N` int array.
   *
   * In broadcasting Python code, this function is equivalent to the following:
   *
   *     idxs_ = idxs + idx_scal_shift
   *     idxs_ += idx_vec_shift_coeff * idx_vec_shift
   *     dst = src[idxs_]
   */
  void subtensor1(float *dst, const float *src, const float *idxs, int src_N,
      int N, int D, float idx_scal_shift, float idx_scal_mul,
      float idx_vec_shift_coeff, float *idx_vec_shift);
  __global__ void k_subtensor1(float *dst, const float *src, const float *idxs,
      int src_N, int N, int D, float idx_scal_shift, float idx_scal_mul,
      float idx_vec_shift_coeff, float *idx_vec_shift);

  /**
   * Write an int scalar into a subtensor range.
   *
   *     dst[idxs + idx_scal_shift + idx_vec_shift_coeff * idx_vec_shift] = src
   */
  void set_subtensor1i_s(float *dst, float src, const float *idxs, int N,
          float idx_scal_shift, float idx_vec_shift_coeff, float *idx_vec_shift);
  __global__ void k_set_subtensor1i_s(float *dst, float src, const float *idxs,
          int N, float idx_scal_shift, float idx_vec_shift_coeff,
          float *idx_vec_shift);

  /**
   * Switch over the rows of two matrices using a mask.
   *
   *     dst = T.switch(mask, ift, iff)
   *
   * where `ift`, `iff` are `N * D` matrices, and `mask` is an `N`-dimensional
   * vector.
   */
  void switch_m(float *dst, const float *mask, const float *ift,
      const float *iff, int N, int D);
  __global__ void k_switch_m(float *dst, const float *mask, const float *ift,
      const float *iff, int N, int D);

}

#endif

#ifndef _kernels_
#define _kernels_

#include <cstdlib>

#include <cuda_runtime.h>


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
  void addi_vv(float *v1, const float *v2, float v1_coeff, float v2_coeff,
      int N);

  /**
   * Retrieve a subset of `N` rows from the contiguous `M * D` matrix `src`
   * and write them to `dst` (`M >= N`). `dst` should be large enough to hold
   * the `N * D` float result. `idxs` should be a length-`N` int array.
   *
   * In broadcasting Python code, this function is equivalent to the following:
   *
   *     idxs_ = idxs + idx_scal_shift
   *     idxs_ += idx_vec_shift_coeff * idx_vec_shift
   *     dst = src[idxs_]
   */
  void subtensor1(float *dst, const float *src, const float *idxs, int N, int D,
      float idx_scal_shift, float idx_vec_shift_coeff, float *idx_vec_shift);
  __global__ void k_subtensor1(float *dst, const float *src, const float *idxs,
      int N, int D, float idx_scal_shift, float idx_vec_shift_coeff,
      float *idx_vec_shift);

  /**
   * Write an int scalar into a subtensor range.
   *
   *     dst[idxs + idx_scal_shift + idx_vec_shift_coeff * idx_vec_shift] = src
   */
  void set_subtensor1i_s(float *dst, int src, const float *idxs, int N,
          float idx_scal_shift, float idx_vec_shift_coeff, float *idx_vec_shift);
  __global__ void set_subtensor1i_s(float *dst, int src, const float *idxs, int N,
          float idx_scal_shift, float idx_vec_shift_coeff, float *idx_vec_shift);

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

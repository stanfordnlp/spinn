#ifndef _blocks_
#define _blocks_

#include <assert.h>
#include <iostream>
#include <fstream>

#include "kernels.cuh"
#include "util.h"

using namespace std;


// Forward declare wrapper classes
class mat;
class vec;


class mat {
  /*
   * Super-thin matrix wrapper around CUDA + CUBLAS.
   */

  public:
    mat(size_t m, size_t n);
    mat(size_t m, size_t n, float *data_);
    mat(size_t m, size_t n, ifstream& file);

    // out = mul(this, other)
    mat& mul(const mat& other, mat& out) const;
    // out += mul(this, other)
    mat& mul_inc(const mat& other, mat& out) const;

    mat& addi(const mat& other, float coeff);
    mat& addi(const vec& other, float coeff);

    mat& muli(const mat& other, float coeff);
    mat& muli(const vec& other, float coeff);
    mat& muli(float scalar);

    mat& divi(const mat& other, float coeff);
    mat& divi(const vec& other, float coeff);

    void relui();

    mat slice1(size_t i) const;
    vec slice1_vec(size_t i) const;

    /**
     * Retrieve a subset of `N` rows from the contiguous matrix `src`
     * and write them to `dst` (`M >= N`). `dst` should be large enough to hold
     * the `N * D` float result. `idxs` should be a length-`N` int array.
     *
     * In broadcasting Python code, this function is equivalent to the following:
     *
     *     idxs_ = idxs * idx_scal_mul + idx_scal_shift
     *     idxs_ += idx_vec_shift_coeff * idx_vec_shift
     *     dst = this[idxs_]
     */
    void subtensor1(mat& dst, const vec& idxs,
        float idx_scal_shift, float idx_scal_mul, float idx_vec_shift_coeff,
        const vec* idx_vec_shift) const;

    /**
     * Write an int scalar into a subtensor range.
     *
     *     this[idxs + idx_scal_shift + idx_vec_shift_coeff * idx_vec_shift] = s
     */
    void set_subtensor1i_s(float s, const vec& idxs,
        float idx_scal_shift, float idx_vec_shift_coeff, const vec* idx_vec_shift);

    void zero();

    ~mat();

    size_t m, n;

    float *data;

};


class vec : public mat {

  public:

    vec(size_t n) : mat(n, 1) {};
    vec(size_t n, float *data_) : mat(n, 1, data) {};
    vec(size_t n, ifstream& file) : mat(n, 1, file) {};

    void addi(const vec& other, float coeff);

    ~vec();

};


typedef struct BatchNormedMLPParameters {
  mat W;
  vec b;

  vec bn_g;
  vec bn_b;
  vec bn_ts;
  vec bn_tm;
} BatchNormedMLPParameters;

void batch_normed_mlp(BatchNormedMLPParameters params, const mat& inp, mat& out);

void xw_plus_b(const mat& W, const mat& b, const mat& x, mat& out);


#endif

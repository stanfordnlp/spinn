#include "blocks.h"
namespace k = kernels;

#include <cuda_runtime.h>
#include "cublas_v2.h"


mat::mat(size_t m, size_t n) : m(m), n(n) {
  cudaMalloc(&data, m * n * sizeof(float));
}

mat::mat(size_t m, size_t n, float *data) : m(m), n(n), data(data) {}

mat::mat(size_t m, size_t n, ifstream& file) : m(m), n(n) {
  cudaMalloc(&data, m * n * sizeof(float));
  float *h_data = (float *) malloc(m * n * sizeof(float));
  float x;
  for (int i = 0; i < m * n; i++) {
    file >> x;
    h_data[i] = x;
  }

  cudaMemcpy(data, h_data, m * n * sizeof(float), cudaMemcpyHostToDevice);
  free(h_data);
}


mat& mat::mul(const mat& other, mat& out) const {
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, m, &alpha, other.data, n, data, n, &beta, out.data, n);
  return out;
}

mat& mat::mul_inc(const mat& other, mat& out) const {
  float alpha = 1.0f;
  float beta = 1.0f;
  cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, m, &alpha, other.data, n, data, n, &beta, out.data, n);
  return out;
}


mat& mat::addi(const mat& other, float coeff) { assert(false); return *this; }


mat& mat::addi(const vec& other, float coeff) {
  k::addi_mv(data, other.data, coeff, m, n);
  return *this;
}


mat& mat::muli(const mat& other, float coeff) { assert(false); return *this; }


mat& mat::muli(const vec& other, float coeff) {
  k::muli_mv(data, other.data, coeff, m, n);
  return *this;
}


mat& mat::muli(float scalar) { assert(false); return *this; }


mat& mat::divi(const mat& other, float coeff) { assert(false); return *this; }


mat& mat::divi(const vec& other, float coeff) {
  k::divi_mv(data, other.data, coeff, m, n);
  return *this;
}


void mat::relui() {
  k::relu(data, m, n);
}


mat mat::slice1(size_t i) const {
  assert(i < m);

  mat ret(m - i, n, &data[i * n]);
  return ret;
}

vec mat::slice1_vec(size_t i) const {
  // Slice and pretend the rest of the data below doesn't exist
  assert(i < m);
  vec ret(n, &data[i * n]);
  return ret;
}


void mat::subtensor1(mat& dst, const vec& idxs, float idx_scal_shift,
    float idx_scal_mul, float idx_vec_shift_coeff, const vec* idx_vec_shift) const {
  float *idx_vec_shift_ = idx_vec_shift == NULL ? NULL : idx_vec_shift->data;
  k::subtensor1(dst.data, data, idxs.data, m, idxs.m, n, idx_scal_shift,
      idx_scal_mul, idx_vec_shift_coeff, idx_vec_shift_);
}


void mat::set_subtensor1i_s(float s, const vec& idxs, float idx_scal_shift,
    float idx_vec_shift_coeff, const vec* idx_vec_shift) {
  float *idx_vec_shift_ = idx_vec_shift == NULL ? NULL : idx_vec_shift->data;
  k::set_subtensor1i_s(data, s, idxs.data, idxs.m, idx_scal_shift,
      idx_vec_shift_coeff, idx_vec_shift_);
}


void mat::zero() {
  cudaMemset(data, 0, m * n * sizeof(float));
}


mat::~mat() {
  cudaFree(data);
}


void vec::addi(const vec& other, float coeff) {
  assert(m == other.m);
  cublasSaxpy(CUBLAS_HANDLE::getInstance(), m, &coeff, other.data, 1, data, 1);
}


vec::~vec() {
  cudaFree(data);
}


void batch_normed_mlp(BatchNormedMLPParameters params, const mat& inp,
                      mat& out) {

  xw_plus_b(params.W, params.b, inp, out);

  // relu
  out.relui();

  // batch-norm
  // - mean
  out.addi(params.bn_tm, -1.0);
  // * gamma
  out.muli(params.bn_g, 1.0);
  // / std
  out.divi(params.bn_ts, 1.0);
  // + beta
  out.addi(params.bn_b, 1.0);

}


void xw_plus_b(const mat& W, const mat& b, const mat& x, mat& out) {

  W.mul(x, out);

  // out += b
  out.addi(b, 1.0);

}


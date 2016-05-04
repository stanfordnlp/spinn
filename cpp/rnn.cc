/**
 * Implements a basic RNN feedforward (ReLU activation) for speed comparison.
 *
 * You can run this code with the script in `bin/rnntest.cc`. See the README in
 * the root directory of this project for full usage instructions.
 */


#include "rnn.h"
using namespace std;
namespace k = kernels;


RNN::RNN(ModelSpec spec, RNNParameters params, cublasHandle_t handle)
  : SequenceModel(spec), params(params), handle(handle) {

  // Pre-allocate inputs.
  cudaMalloc(&X_indices, spec.batch_size * spec.seq_length * sizeof(float));
  cudaMalloc(&X, spec.batch_size * spec.seq_length * spec.model_dim * sizeof(float));

  // Pre-allocate temporary containers.
  cudaMalloc(&odd_output, spec.batch_size * spec.model_dim * sizeof(float));
  cudaMalloc(&even_output, spec.batch_size * spec.model_dim * sizeof(float));

  output = spec.seq_length % 2 == 0 ? even_output : odd_output;

}


RNN::~RNN() {

  cout << "!!!!!!!!!!" << endl;
  cout << "RNN dying!" << endl;
  cout << "!!!!!!!!!!" << endl;

  cudaFree(X_indices);
  cudaFree(X);

}


void RNN::forward() {

  // First timestep will read from odd output slot. Make sure it sees an empty
  // (zero) state.
  cudaMemset(odd_output, 0, spec.model_dim * spec.batch_size * sizeof(float));

  for (int t = 0; t < spec.seq_length; t++) {
    step(t);
#if DEBUG
    cout << endl << "======================" << endl << endl;
#endif
  }

  // TODO: Don't need to sync here. Could have the client establish a lock on
  // results and simultaneously begin the next batch + copy out results
  cudaDeviceSynchronize();

}


void RNN::step(int t) {

  const float *X_t = &X[t * spec.word_embedding_dim * spec.batch_size];
#if DEBUG
  cout << "X_t " << t << endl;
  print_device_matrix(X_t, spec.model_dim, spec.batch_size);
#endif

  const float *state;
  float *output;
  if (t % 2 == 0) {
    // t is even -- read from output at previous odd timestep, and write into
    // even slot
    state = odd_output;
    output = even_output;
  } else {
    // t is odd -- read from output at previous even timestep, and write into
    // odd slot
    state = even_output;
    output = odd_output;
  }

  recurrence(state, X_t, output);

}


void RNN::recurrence(const float *state, const float *input, float *output) {

  // out = U(state)
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, spec.model_dim, spec.batch_size,
      spec.model_dim, &alpha, params.U, spec.model_dim, state, spec.model_dim,
      &beta, output, spec.model_dim);
  // out += W(input)
  float beta2 = 1.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, spec.model_dim, spec.batch_size,
      spec.model_dim, &alpha, params.W, spec.model_dim, input, spec.model_dim,
      &beta2, output, spec.model_dim);

  // out += b
  k::addi_mv(output, params.b, 1.0, spec.model_dim, spec.batch_size);

  k::relu(output, spec.model_dim, spec.batch_size);

#if DEBUG
  cout << "state" << endl;
  print_device_matrix(output, spec.model_dim, spec.batch_size);
#endif

}

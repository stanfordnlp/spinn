#ifndef _thin_stack_
#define _thin_stack_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "sequence-model.h"
#include "util.h"

#include "kernels.cuh"


typedef struct RNNParameters {
  float *U; // hidden-to-hidden
  float *W; // input-to-hidden
  float *b;
} RNNParameters;

class RNN : public SequenceModel {
  public:
    RNN(ModelSpec spec, RNNParameters params, cublasHandle_t handle);
    ~RNN();

    RNNParameters params;
    cublasHandle_t handle;

    void forward();

    float *output;

  private:

    void step(int t);

    void recurrence(const float *state, const float *input, float *output);

    // Containers for temporary (per-step) data
    // RNN feedforward need only maintain two state cells. Just read from one
    // and write to the other!
    float *odd_output, *even_output;

};

#endif

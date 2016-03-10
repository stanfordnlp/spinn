#ifndef _thin_stack_
#define _thin_stack_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "util.h"

#include "kernels.cuh"


typedef struct ThinStackParameters {
  float *embeddings;
  float *project_W;
  float *project_b;
  float *buffer_bn_ts;
  float *buffer_bn_tm;
  float *tracking_W_inp;
  float *tracking_W_hid;
  float *tracking_b;
  float *compose_W_l;
  float *compose_W_r;
  float *compose_W_ext;
  float *compose_b;
} ThinStackParameters;

class ThinStack {
  public:
    /**
     * Constructs a new `ThinStack`.
     */
    ThinStack(ModelSpec spec, ThinStackParameters params);

    ~ThinStack();

    void forward();

    // Model inputs
    float *X;
    int *transitions;
    float *y;

    float *stack;

  private:

    void step(int t);
    void recurrence();
    void zero();

    ModelSpec spec;
    ThinStackParameters params;

    size_t stack_size;

    size_t stack_total_size;
    size_t queue_total_size;
    size_t cursors_total_size;

    // Containers for temporary (per-step) data
    int *buffer_top_idxs_t;
    float *buffer_top_t;
    int *stack_1_ptrs;
    float *stack_1_t;
    int *stack_2_ptrs;
    float *stack_2_t;
    float *push_output;
    float *merge_output;

    // Per-step accumulators
    int *buffer_cur_t;

    float *buffer;
    float *queue;
    float *cursors;

};

#endif

#ifndef _thin_stack_
#define _thin_stack_

#include "util.h"

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

    void forward();

    // Model inputs
    float *X;
    float *y;

    float *stack;

  private:

    void zero();

    ModelSpec spec;
    ThinStackParameters params;

    size_t stack_size;

    size_t stack_total_size;
    size_t queue_total_size;
    size_t cursors_total_size;

    float *queue;
    float *cursors;

};

#endif

#ifndef _thin_stack_
#define _thin_stack_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "sequence-model.h"
#include "util.h"

#include "kernels.cuh"


struct ThinStackSpec : public SequenceModelSpec {
  size_t tracking_lstm_dim;

  ThinStackSpec(size_t model_dim, size_t word_embedding_dim, size_t batch_size,
                size_t vocab_size, size_t seq_length, size_t model_visible_dim,
                size_t tracking_lstm_dim)
      : SequenceModelSpec(model_dim, word_embedding_dim, batch_size,
                          vocab_size, seq_length, model_visible_dim),
        tracking_lstm_dim(tracking_lstm_dim) {};
};


typedef struct ThinStackParameters {
  float *tracking_W_inp;
  float *tracking_W_hid;
  float *tracking_b;
  float *compose_W_l;
  float *compose_W_r;
  float *compose_W_ext;
  float *compose_b;
} ThinStackParameters;

class ThinStack : public SequenceModel {
  public:
    /**
     * Constructs a new `ThinStack`.
     */
    ThinStack(ThinStackSpec spec, ThinStackParameters params,
            cublasHandle_t handle);

    ~ThinStack();

    ThinStackParameters params;
    cublasHandle_t handle;

    void forward();

    float *transitions;

    float *stack;

  private:

    void step(int t);

    // Reset internal storage. Must be run before beginning a sequence
    // feedforward.
    void reset();

    void recurrence(const float *stack_1_t, const float *stack_2_t,
            const float *buffer_top_t);
    void mask_and_update_stack(const float *push_value,
            const float *merge_value, const float *transitions, int t);
    void mask_and_update_cursors(float *cursors, const float *transitions,
                                 int t);
    void update_buffer_cur(float *buffer_cur_t, float *transitions, int t);

    void init_helpers();
    void free_helpers();

    size_t stack_size;

    size_t stack_total_size;
    size_t buffer_total_size;
    size_t queue_total_size;
    size_t cursors_total_size;

    // Containers for temporary (per-step) data
    float *buffer_top_idxs_t;
    float *buffer_top_t;
    float *stack_1_ptrs;
    float *stack_1_t;
    float *stack_2_ptrs;
    float *stack_2_t;
    float *push_output;
    float *merge_output;

    // Per-step accumulators
    float *buffer_cur_t;

    // Dumb helpers
    float *batch_ones;
    float *batch_range;

    // `model_dim * (batch_size * seq_length)`
    // `seq_length`-many `model_dim * batch_size` matrices, flattened into one.
    float *buffer;
    float *queue;
    float *cursors;

};

#endif

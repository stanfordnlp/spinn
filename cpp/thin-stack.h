#ifndef _thin_stack_
#define _thin_stack_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "blocks.h"
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
  mat tracking_W_inp;
  mat tracking_W_hid;
  mat tracking_b;
  mat compose_W_l;
  mat compose_W_r;
  mat compose_W_ext;
  mat compose_b;
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

    mat transitions;
    mat stack;

    // Return the final representations at the top of the stack after running a
    // feedforward.
    // Matrix of dimension spec.model_dim * spec.batch_size
    mat final_representations();

  private:

    void step(int t);

    // Reset internal storage. Must be run before beginning a sequence
    // feedforward.
    void reset();

    void recurrence(const mat& stack_1_t, const mat& stack_2_t,
                    const mat& buffer_top_t);
    void mask_and_update_stack(const mat& push_value, const mat& merge_value,
            const vec& transitions_t, int t);
    void mask_and_update_cursors(vec& cursors, const vec& transitions_t);
    void update_buffer_cur(vec& buffer_cur_t, const vec& transitions_t);

    void init_helpers();

    size_t stack_size;

    size_t stack_total_size;
    size_t buffer_total_size;
    size_t queue_total_size;
    size_t cursors_total_size;

    // Containers for temporary (per-step) data
    vec buffer_top_idxs_t;
    mat buffer_top_t;
    vec stack_1_ptrs;
    mat stack_1_t;
    vec stack_2_ptrs;
    mat stack_2_t;
    mat push_output;
    mat merge_output;

    // Per-step accumulators
    vec buffer_cur_t;

    // Dumb helpers
    vec batch_ones;
    vec batch_range;

    mat queue;
    vec cursors;

};

#endif

#include "thin-stack.h"
using namespace std;

ThinStack::ThinStack(ModelSpec spec, ThinStackParameters params)
  : spec(spec), params(params), stack_size(spec.seq_length) {

  stack_total_size = (stack_size * spec.batch_size) * spec.model_dim;
  buffer_total_size = spec.batch_size * spec.seq_length * spec.model_dim;
  queue_total_size = spec.batch_size * spec.seq_length;
  cursors_total_size = spec.batch_size;

  // Pre-allocate auxiliary data structures.
  cout << stack << endl;
  cudaMalloc(&stack, stack_total_size * sizeof(float));
  cout << stack << endl;
  cudaMalloc(&queue, queue_total_size * sizeof(float));
  cudaMalloc(&cursors, cursors_total_size * sizeof(float));
  cudaMalloc(&buffer, buffer_total_size * sizeof(int));

  // Pre-allocate temporary containers.
  cudaMalloc(&buffer_top_idxs_t, spec.batch_size * sizeof(int));
  cudaMalloc(&buffer_top_t, spec.batch_size * spec.model_dim * sizeof(float));
  cudaMalloc(&stack_1_ptrs, spec.batch_size * sizeof(int));
  cudaMalloc(&stack_2_ptrs, spec.batch_size * sizeof(int));
  cudaMalloc(&push_output, spec.batch_size * spec.model_dim * sizeof(int));
  cudaMalloc(&merge_output, spec.batch_size * spec.model_dim * sizeof(int));

  // Pre-allocate accumulators.
  cudaMalloc(&buffer_cur_t, spec.batch_size * sizeof(float));

}


ThinStack::~ThinStack() {

  cudaFree(stack);
  cudaFree(queue);
  cudaFree(cursors);
  cudaFree(buffer);

  cudaFree(buffer_top_idxs_t);
  cudaFree(buffer_top_t);
  cudaFree(stack_1_ptrs);
  cudaFree(stack_2_ptrs);
  cudaFree(push_output);
  cudaFree(merge_output);

  cudaFree(buffer_cur_t);

}


ThinStack::forward() {

  for (int t = 0; t < spec.seq_length; t++) {
    step(t);
  }

}


ThinStack::step(int t) {

  // Extract top buffer values.
  // TODO subtensor with idxs buffer_cur_t + 0 * 1 + buffer_shift
  // == buffer_cur_t * 1 + (batch_range * seq_length)
  subtensor1(buffer_top_t, buffer, buffer_cur_t, 0, 1, spec.seq_length)

  // stack_1_ptrs = (t - 1) * batch_size + batch_range * 1
  add_sv(stack_1_ptrs, (t - 1) * spec.batch_size, 1)

  // stack_2_ptrs = (cursors - 1) + batch_range * seq_length
  subtensor1(stack_2_ptrs, queue, cursors, -1, 1, spec.seq_length)
  // stack_2_ptrs = stack_2_ptrs * batch_size + batch_range * 1
  addi_vv(stack_2_ptrs, batch_size, 1)

  // Run recurrence, which writes into `push_output`, `merge_output`.
  recurrence(stack_1_ptrs, stack_2_ptrs, buffer_top_t);

  // Write in the next stack top.
  // NB: destroying stack_1_ptrs
  add_sv(stack_1_ptrs, t * spec.batch_size, 1);
  mask_and_update_stack(stack_1_ptrs, push_output, merge_output,
                        transitions, t);

  // cursors += cursors + 1 - 2*mask
  mask_and_update_cursors(cursors, transitions, t);

  // queue[cursors + 0 + batch_range * spec.seq_length] = t
  set_subtensor1i_s(queue, cursors, 0, spec.spec_length);

  update_buffer_cur(buffer_cur_t, transitions, t);

}


ThinStack::zero() {
  cudaMemset(stack, 0, stack_total_size * sizeof(float));
  cudaMemset(queue, 0, queue_total_size * sizeof(float));
  cudaMemset(cursors, 0, cursors_total_size * sizeof(float));

  cudaMemset(buffer_cur_t, 0, spec.batch_size * sizeof(float));
}

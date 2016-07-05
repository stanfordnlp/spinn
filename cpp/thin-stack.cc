/**
 * This file contains a C++/CUDA implementation of the thin-stack algorithm.
 * Thin-stack computes the same function as a vanilla recursive neural network
 * and as the SPINN-PI-NT model.
 *
 * The SPINN model and the thin-stack algorithm are described in our paper:
 *
 *     A Fast Unified Model for Sentence Parsing and Understanding.
 *     Samuel R. Bowman, Jon Gauthier, Abhinav Rastogi, Raghav Gupta,
 *     Christopher D. Manning, and Christopher Potts. arXiv March 2016.
 *     http://arxiv.org/abs/1603.06021
 *
 * The exact model implemented here is a recursive neural network (equivalent
 * to SPINN-PI-NT) with ReLU activations. It has been verified to compute the
 * exact same function of its inputs as a recursive neural network.
 *
 * You can execute this code using the script `bin/stacktest`. See the README
 * in the root directory of this project for more usage instructions.
 */


#include "thin-stack.h"
using namespace std;
namespace k = kernels;


ThinStack::ThinStack(ThinStackSpec spec, ThinStackParameters params,
    cublasHandle_t handle)
  : SequenceModel(spec), params(params), stack_size(spec.seq_length),
    handle(handle),

    // Allocate inputs.
    transitions(spec.batch_size, spec.seq_length),

    // Allocate auxiliary data structures.
    stack(stack_size * spec.batch_size, spec.model_dim),
    queue(spec.batch_size, spec.seq_length),
    cursors(spec.batch_size),

    // Allocate temporary containers.
    buffer_top_idxs_t(spec.batch_size),
    buffer_top_t(spec.model_dim, spec.batch_size),
    stack_1_ptrs(spec.batch_size),
    stack_1_t(spec.model_dim, spec.batch_size),
    stack_2_ptrs(spec.batch_size),
    stack_2_t(spec.model_dim, spec.batch_size),
    push_output(spec.model_dim, spec.batch_size),
    merge_output(spec.model_dim, spec.batch_size),

    // Allocate accumulators.
    buffer_cur_t(spec.batch_size),

    // Allocate helpers.
    batch_ones(spec.batch_size), batch_range(spec.batch_size) {

  init_helpers();

}


void ThinStack::init_helpers() {
  float h_batch_ones[spec.batch_size];
  float h_batch_range[spec.batch_size];
  for (int i = 0; i < spec.batch_size; i++) {
    h_batch_ones[i] = 1.0f;
    h_batch_range[i] = (float) i;
  }

  cudaMemcpy(batch_range.data, h_batch_range, spec.batch_size * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(batch_ones.data, h_batch_ones, spec.batch_size * sizeof(float),
      cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}


ThinStack::~ThinStack() {

  cout << "!!!!!!!!!!!!!!!!" << endl;
  cout << "ThinStack dying!" << endl;
  cout << "!!!!!!!!!!!!!!!!" << endl;

}


void ThinStack::forward() {

  // TODO embedding projection
  reset();
  cudaDeviceSynchronize();

  for (int t = 0; t < spec.seq_length; t++) {
    step(t);
#if DEBUG
    cout << endl << "======================" << endl << endl;
#endif
  }

  // TODO: Don't need to sync here. Could have the client establish a lock on
  // results and simultaneously begin the next batch + copy out results
  cudaDeviceSynchronize();

#if DEBUG
  cout << "final" << endl;
  print_device_matrix(stack, spec.model_dim, spec.batch_size * spec.seq_length);
#endif

}


void ThinStack::step(int t) {

  vec transitions_t = transitions.slice1_vec(t);
#if DEBUG
  cout << "transitions " << t << endl;
  print_device_matrix(transitions_t, 1, spec.batch_size);
#endif

  // buffer_top = buffer[buffer_cur_t * batch_size + batch_range]
  X.subtensor1(buffer_top_t, buffer_cur_t, 0.0f, spec.batch_size, 1.0f, &batch_range);
#if DEBUG
  cout << "buffer_top after:" << endl;
  print_device_matrix(buffer_top_t, spec.model_dim, spec.batch_size);
#endif

  // stack_2_ptrs = (cursors - 1) + batch_range * seq_length
  queue.subtensor1(stack_2_ptrs, cursors, -1.0f, 1.0f, spec.seq_length, &batch_range);
#if DEBUG
  cout << "stack_2_ptrs #1" << endl;
  print_device_matrix(stack_2_ptrs, 1, spec.batch_size);
#endif

  // stack_2_ptrs = stack_2_ptrs * batch_size + batch_range * 1
  stack_2_ptrs.muli(spec.batch_size);
  stack_2_ptrs.addi(batch_range, 1.0f);
#if DEBUG
  cout << "stack_2_ptrs" << endl;
  print_device_matrix(stack_2_ptrs, 1, spec.batch_size);
#endif

  // stack_1, stack_2
  // stack_1_t = stack[batch_range + (t - 1) * spec.batch_size]
  stack.subtensor1(stack_1_t, batch_range, (float) (t - 1) * spec.batch_size,
      1.0f, 0.0f, NULL);

  stack.subtensor1(stack_2_t, stack_2_ptrs, 0.0f, 1.0f, 0.0f, NULL);

  // Run recurrence, which writes into `push_output`, `merge_output`.
  recurrence(stack_1_t, stack_2_t, buffer_top_t);

  // Write in the next stack top.
  mask_and_update_stack(buffer_top_t, merge_output, transitions_t, t);

  mask_and_update_cursors(cursors, transitions_t);
#if DEBUG
  cout << "cursors after" << endl;
  print_device_matrix(cursors, 1, spec.batch_size);
#endif


  // queue[cursors + 0 + batch_range * spec.seq_length] = t
  queue.set_subtensor1i_s(t, cursors, 0.0f, spec.seq_length, &batch_range);
#if DEBUG
  cout << "queue after" << endl;
  print_device_matrix(queue, 1, spec.seq_length * spec.batch_size);
#endif

  // buffer_cur += (1 - transitions)
  update_buffer_cur(buffer_cur_t, transitions_t);
#if DEBUG
  cout << "buffer cur after" << endl;
  print_device_matrix(buffer_cur_t, 1, spec.batch_size);
#endif

}


void ThinStack::recurrence(const mat& stack_1_t, const mat& stack_2_t,
    const mat& buffer_top_t) {

#if DEBUG
  cout << "left child:" << endl;
  print_device_matrix(stack_2_t, spec.model_dim, spec.batch_size);

  cout << "right child:" << endl;
  print_device_matrix(stack_1_t, spec.model_dim, spec.batch_size);
#endif

  // merge_out = W_l l
  params.compose_W_l.mul(stack_2_t, merge_output);
  // merge_out += W_r r
  params.compose_W_r.mul_inc(stack_1_t, merge_output);

  // merge_out += b
  merge_output.addi(params.compose_b, 1.0f);

  merge_output.relui();

}


void ThinStack::mask_and_update_stack(const mat& push_value, const mat& merge_value,
    const vec& transitions_t, int t) {

  // Find start position of write destination (next-top corresponding to
  // timestep `t`).
  int stack_offset = t * spec.batch_size * spec.model_dim;

#if DEBUG
  cout << "merge value:" << endl;
  print_device_matrix(merge_value, spec.model_dim, spec.batch_size);
  cout << "push value:" << endl;
  print_device_matrix(push_value, spec.model_dim, spec.batch_size);
#endif

  k::switch_m(&stack.data[stack_offset], transitions.data, merge_value.data, push_value.data,
              spec.batch_size, spec.model_dim);

#if DEBUG
  cout << "stack top t (offset " << stack_offset << "):" << endl;
  print_device_matrix(&stack[stack_offset], spec.model_dim, spec.batch_size);
#endif

}


void ThinStack::mask_and_update_cursors(vec& cursors, const vec& transitions_t) {

  // cursors += 1
  cursors.addi(batch_ones, 1.0f);

  // cursors -= 2*transitions
  cursors.addi(transitions_t, -2.0f);

}


void ThinStack::update_buffer_cur(vec& buffer_cur_t, const vec& transitions_t) {

  // buffer_cur += 1
  buffer_cur_t.addi(batch_ones, 1.0f);

  // buffer_cur -= transitions
  buffer_cur_t.addi(transitions_t, -1.0f);

}


void ThinStack::reset() {
  // TODO: Technically these don't need to be explicitly zeroed out before
  // every feedforward. They just get overwritten and their bad values are
  // never used, provided that the feedforward uses a valid transition
  // sequence.
  stack.zero();
  queue.zero();

  cursors.zero();
  cursors.addi(batch_ones, -1.0f);
  float alpha = -1.0f;

  buffer_cur_t.zero();
}


mat ThinStack::final_representations() {
  return stack.slice1((spec.seq_length - 1) * spec.model_dim);
}

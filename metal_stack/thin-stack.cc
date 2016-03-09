#include "thin-stack.h"
using namespace std;

ThinStack::ThinStack(ModelSpec spec, ThinStackParameters params)
  : spec(spec), params(params), stack_size(spec.seq_length) {

  stack_total_size = (stack_size * spec.batch_size) * spec.model_dim;
  queue_total_size = spec.batch_size * spec.seq_length;
  cursors_total_size = spec.batch_size;

  // Pre-allocate auxiliary data structures.
  cout << stack << endl;
  cudaMalloc(&stack, stack_total_size * sizeof(float));
  cout << stack << endl;
  cudaMalloc(&queue, queue_total_size * sizeof(float));
  cudaMalloc(&cursors, cursors_total_size * sizeof(float));

}


ThinStack::forward() {
  // TODO
}


ThinStack::zero() {
  cudaMemset(stack, 0, stack_total_size * sizeof(float));
  cudaMemset(queue, 0, queue_total_size * sizeof(float));
  cudaMemset(cursors, 0, cursors_total_size * sizeof(float));
}

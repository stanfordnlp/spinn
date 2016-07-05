#include "spinn.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;


Spinn::Spinn(SpinnSpec spec, SpinnParameters params, cublasHandle_t handle)
  : spec(spec), params(params), handle(handle),
    ts(spec.ts_spec, params.ts_params, handle) {

  // TODO: will be necessary to have own storage once we are batch-norming
  /* // Allocate input */
  /* cudaMalloc(&X, spec.batch_size * spec.seq_length * spec.model_dim * sizeof(float)); */
  /* cudaMalloc(&transitions, spec.batch_size * spec.seq_length * sizeof(float)); */
  X = ts.X;
  transitions = ts.transitions;

  // Allocate sentence combination storage
  for (int i = 0; i < spec.num_combination_layers; i++)
    cudaMalloc(&sentence_combinations[i],
        spec.combination_layer_dim * spec.batch_size * sizeof(float));

  cudaMalloc(&logits, spec.num_classes * spec.batch_size * sizeof(float));

}


int main() {
  // TODO parse args to build spec
  SpinnSpec spec;
  string model_path;

  SpinnParameters params = load_params(spec, model_path);

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "CUBLAS initialization failed" << endl;
    return 1;
  }

  ThinStack ts(spec.ts_spec, params.ts_params, handle);
  ts.forward();

  // TODO combination features

  float *representation = ts.final_representations();
  int m = spec.ts_spec.model_dim;
  // Output
  float *next_representation;
  cudaMalloc(&next_representation, spec.batch_size * spec.combination_layer_dim * sizeof(float));

  for (int i = 0; i < num_combination_layers; i++) {
    BatchNormedMLPParameters mlp_i = params.mlp_params[i];
    batch_normed_mlp(m, spec.combination_layer_dim, mlp_i, representation,
        next_representation, handle);

    std::swap(representation, next_representation);
    m = spec.combination_layer_dim;
  }
  representation = num_combination_layers % 2 == 0
    ? representation : next_representation;

  // TODO should be allocated class-level, once we have a class
  float *classifier_output;
  cudaMalloc(&classifier_output, spec.num_classes * spec.batch_size * sizeof(float));
  xw_plus_b(spec.combination_layer_dim, spec.batch_size, params.classifier_W,
      params.classifier_b, representation, classifier_output, handle);

}

#include "spinn.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;


Spinn::Spinn(SpinnSpec spec, SpinnParameters params, cublasHandle_t handle)
  : spec(spec), params(params), handle(handle),
    ts(spec.ts_spec, params.ts_params, handle),

    X(spec.model_dim, spec.batch_size * spec.seq_length),
    y(spec.num_classes, spec.batch_size),
    transitions(spec.batch_size, spec.seq_length),
    logits(spec.num_classes, spec.batch_size) {

  // TODO: will be necessary to have own storage once we are batch-norming
  /* // Allocate input */
  /* cudaMalloc(&X, spec.batch_size * spec.seq_length * spec.model_dim * sizeof(float)); */
  /* cudaMalloc(&transitions, spec.batch_size * spec.seq_length * sizeof(float)); */
  /* X = ts.X; */
  /* transitions = ts.transitions; */

  // Allocate sentence combination storage
  for (int i = 0; i < spec.num_combination_layers; i++) {
    mat sentence_combination_i(spec.combination_layer_dim, spec.batch_size);
    sentence_combinations.push_back(sentence_combination_i);
  }

}


void Spinn::forward() {

  ts.forward();

  // TODO sentence-level batch-norm
  // TODO combination features

  mat representation = ts.final_representations();
  int m = spec.model_dim;
  for (int i = 0; i < spec.num_combination_layers; i++) {
    BatchNormedMLPParameters mlp_i = params.mlp_params[i];
    batch_normed_mlp(mlp_i, representation, sentence_combinations[i]);

    representation = sentence_combinations[i];
    m = spec.combination_layer_dim;
  }

  xw_plus_b(params.classifier_W, params.classifier_b, representation, logits);

}


int main() {
/*   // TODO parse args to build spec */
/*   ThinStackSpec ts_spec(300, 300, 64, 1000, 30, 300, 0); */
/*   SpinnSpec spec(3, 2, 1024, ts_spec, false, false); */
/*   string model_path; */

/*   SpinnParameters params = load_params(spec, model_path); */

/*   cublasHandle_t handle; */
/*   cublasStatus_t stat = cublasCreate(&handle); */
/*   if (stat != CUBLAS_STATUS_SUCCESS) { */
/*     cout << "CUBLAS initialization failed" << endl; */
/*     return 1; */
/*   } */

/*   Spinn spinn(spec, params, handle); */

}

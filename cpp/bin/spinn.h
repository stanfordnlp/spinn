#ifndef _spinn_
#define _spinn_

#include <cstddef>
#include <fstream>
#include <vector>

#include "blocks.h"
#include "sequence-model.h"
#include "thin-stack.h"
#include "util.h"


struct SpinnSpec {
  size_t num_classes;
  size_t batch_size;
  size_t model_dim;
  size_t seq_length;
  size_t num_combination_layers;
  size_t combination_layer_dim;

  ThinStackSpec ts_spec;

  bool use_difference_feature;
  bool use_product_feature;

  SpinnSpec(size_t num_classes, size_t num_combination_layers,
            size_t combination_layer_dim, ThinStackSpec ts_spec,
            bool use_difference_feature, bool use_product_feature)
      : num_classes(num_classes), num_combination_layers(num_combination_layers),
        combination_layer_dim(combination_layer_dim), ts_spec(ts_spec),
        use_difference_feature(use_difference_feature),
        use_product_feature(use_product_feature),

        batch_size(ts_spec.batch_size), model_dim(ts_spec.model_dim),
        seq_length(ts_spec.seq_length) {};
};

typedef struct SpinnParameters {
  BatchNormedMLPParameters projection;

  ThinStackParameters ts_params;

  // Sentence representation batch-norm parameters
  vec sentence_bn_g;
  vec sentence_bn_b;
  vec sentence_bn_ts;
  vec sentence_bn_tm;

  BatchNormedMLPParameters *mlp_params;

  mat classifier_W;
  vec classifier_b;
} SpinnParameters;


class Spinn {
  /**
   * SPINN classifier which wraps around a thin-stack algorithm instance.
   */

  public:
    Spinn(SpinnSpec spec, SpinnParameters params, cublasHandle_t handle);

    SpinnSpec spec;
    SpinnParameters params;
    cublasHandle_t handle;

    void forward();

    mat X;
    mat y;
    mat transitions;

  private:

    ThinStack ts;

    // Feedforward containers
    vector<mat> sentence_combinations;
    mat logits;

};


/* BatchNormedMLPParameters load_projection_params(int m, int n, ifstream& file) { */
/*   mat W(m, n, file); */
/*   vec b(n, file); */

/*   vec bn_g(n, file); */
/*   vec bn_b(n, file); */
/*   vec bn_ts(n, file); */
/*   vec bn_tm(n, file); */

/*   BatchNormedMLPParameters params = { */
/*     W, b, bn_g, bn_b, bn_ts, bn_tm */
/*   }; */
/*   return params; */
/* } */


/* ThinStackParameters load_ts_params(ThinStackSpec spec, ifstream& file) { */
/*   size_t hidden_dim = spec.model_dim / 2; */
/*   vec tracking_b(spec.tracking_lstm_dim * 4, file); */
/*   mat tracking_W_inp(hidden_dim * 3, spec.model_dim, file); */
/*   mat tracking_W_hid(spec.tracking_lstm_dim, spec.tracking_lstm_dim * 4, file); */

/*   mat compose_W_l(hidden_dim, hidden_dim * 5, file); */
/*   mat compose_W_r(hidden_dim, hidden_dim * 5, file); */
/*   vec compose_b(hidden_dim * 5, file); */

/*   ThinStackParameters params = { */
/*     tracking_W_inp, tracking_W_hid, tracking_b, */
/*     compose_W_l, compose_W_r, NULL, compose_b */
/*   }; */

/*   return params; */
/* } */


/* SpinnParameters load_params(SpinnSpec spec, string path) { */
/*   ifstream file(path); */

/*   BatchNormedMLPParameters projection = load_projection_params( */
/*       spec.ts_spec.word_embedding_dim, spec.ts_spec.model_dim, file); */
/*   ThinStackParameters ts_params = load_ts_params(spec.ts_spec, file); */

/*   size_t sentence_vector_dim = spec.ts_spec.model_dim / 2; */
/*   size_t mlp_input_dim = sentence_vector_dim * 2; */
/*   if (spec.use_difference_feature) mlp_input_dim += sentence_vector_dim; */
/*   if (spec.use_product_feature) mlp_input_dim += sentence_vector_dim; */

/*   vec sentence_bn_g = load_weights_cuda(file, mlp_input_dim); */
/*   vec sentence_bn_b = load_weights_cuda(file, mlp_input_dim); */
/*   vec sentence_bn_ts = load_weights_cuda(file, mlp_input_dim); */
/*   vec sentence_bn_tm = load_weights_cuda(file, mlp_input_dim); */

/*   BatchNormedMLPParameters mlp_params[spec.num_combination_layers]; */
/*   assert(spec.num_combination_layers > 0); // only support w/ combination layers */

/*   mlp_params[0] = load_projection_params(mlp_input_dim, spec.combination_layer_dim, file); */
/*   for (int i = 1; i < spec.num_combination_layers; i++) */
/*     mlp_params[i] = load_projection_params(spec.combination_layer_dim, spec.combination_layer_dim, file); */

/*   mat classifier_W = load_weights_cuda(file, spec.combination_layer_dim * 3); */
/*   vec classifier_b = load_weights_cuda(file, 3); */

/*   SpinnParameters params = { */
/*     projection, ts_params, */
/*     sentence_bn_g, sentence_bn_b, sentence_bn_ts, sentence_bn_tm, */
/*     mlp_params, */
/*     classifier_W, classifier_b */
/*   }; */

/*   return params; */
/* } */


#endif

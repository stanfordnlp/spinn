#ifndef _spinn_
#define _spinn_

#include <cstddef>
#include <fstream>

#include "sequence-model.h"
#include "thin-stack.h"
#include "util.h"


typedef struct SpinnSpec {
  ModelSpec ts_spec;
  size_t num_combination_layers;
  size_t combination_layer_dim;

  bool use_difference_feature;
  bool use_product_feature;
}

typedef struct SpinnParameters {
  BatchNormedMLPParameters projection;

  ThinStackParameters ts_params;

  // Sentence representation batch-norm parameters
  float *sentence_bn_g;
  float *sentence_bn_b;
  float *sentence_bn_ts;
  float *sentence_bn_tm;

  BatchNormedMLPParameters *mlp_params;

  float *classifier_W;
  float *classifier_b;
}


class Spinn {
  /**
   * SPINN classifier which wraps around a thin-stack algorithm instance.
   */

  public:
    Spinn(SpinnSpec spec, SpinnParameters params, cublasHandle_t handle);

    SpinnParameters params;
    cublasHandle_t handle;

    void forward();

    float *X;
    float *y;
    float *transitions;

  private:

    ThinStack ts;

    // Feedforward containers
    float **sentence_combinations;
    float *logits;

}


BatchNormedMLPParameters load_projection_params(int m, int n, ifstream& file) {
  float *W = load_weights_cuda(file, m * n);
  float *b = load_weights_cuda(file, n);

  float *bn_g = load_weights_cuda(file, n);
  float *bn_b = load_weights_cuda(file, n);
  float *bn_ts = load_weights_cuda(file, n);
  float *bn_tm = load_weights_cuda(file, n);

  BatchNormedMLPParameters params = {
    W, b, bn_g, bn_b, bn_ts, bn_tm
  };
  return params;
}


ThinStackParameters load_ts_params(ModelSpec spec, ifstream& file) {
  size_t hidden_dim = spec.model_dim / 2;
  float *tracking_b = load_weights_cuda(file, spec.tracking_lstm_dim * 4);
  float *tracking_W_inp = load_weights_cuda(file, (hidden_dim * 3) * spec.tracking_lstm_dim);
  float *tracking_W_hid = load_weights_cuda(file, spec.tracking_lstm_dim * (spec.tracking_lstm_dim * 4));

  float *compose_W_l = load_weights_cuda(file, hidden_dim * (hidden_dim * 5));
  float *compose_W_r = load_weights_cuda(file, hidden_dim * (hidden_dim * 5));
  float *compose_b = load_weights_cuda(file, hidden_dim * 5);

  ThinStackParameters params = {
    tracking_W_inp, tracking_W_hid, tracking_b,
    compose_W_l, compose_W_r, NULL, compose_b
  };

  return params;
}


SpinnParameters load_params(SpinnSpec spec, string path) {
  ifstream file(path);

  BatchNormedMLPParameters projection = load_projection_params(
      spec.ts_spec.word_embedding_dim, spec.ts_spec.model_dim, file);
  ThinStackParameters ts_params = load_ts_params(spec.ts_spec, file);

  size_t sentence_vector_dim = spec.ts_spec.model_dim / 2;
  size_t mlp_input_dim = sentence_vector_dim * 2;
  if (spec.use_difference_feature) mlp_input_dim += sentence_vector_dim;
  if (spec.use_product_feature) mlp_input_dim += sentence_vector_dim;

  float *sentence_bn_g = load_weights_cuda(file, mlp_input_dim);
  float *sentence_bn_b = load_weights_cuda(file, mlp_input_dim);
  float *sentence_bn_ts = load_weights_cuda(file, mlp_input_dim);
  float *sentence_bn_tm = load_weights_cuda(file, mlp_input_dim);

  BatchNormedMLPParameters mlp_params[spec.num_combination_layers];
  assert(spec.num_combination_layers > 0); // only support w/ combination layers

  mlp_params[0] = load_projection_params(mlp_input_dim, spec.combination_layer_dim, file);
  for (int i = 1; i < spec.num_combination_layers; i++)
    mlp_params[i] = load_projection_params(spec.combination_layer_dim, spec.combination_layer_dim, file);

  float *classifier_W = load_weights_cuda(file, spec.combination_layer_dim * 3);
  float *classifier_b = load_weights_cuda(file, 3);

  SpinnParameters params = {
    projection, ts_params,
    sentence_bn_g, sentence_bn_b, sentence_bn_ts, sentence_bn_tm,
    mlp_params,
    classifier_W, classifier_b
  };

  return params;
}


void destroy_params(ThinStackParameters params) {
  cudaFree(params.tracking_b);
  cudaFree(params.tracking_W_inp);
  cudaFree(params.tracking_W_hid);

  cudaFree(params.compose_W_l);
  cudaFree(params.compose_W_r);
  cudaFree(params.compose_b);
}


void destroy_params(BatchNormedMLPParameters params) {
  cudaFree(params.W);
  cudaFree(params.b);

  cudaFree(params.bn_g);
  cudaFree(params.bn_b);
  cudaFree(params.bn_ts);
  cudaFree(params.bn_tm);
}


void destroy_params(SpinnSpec spec, SpinnParameters params) {
  destroy_params(params.projection);
  destroy_params(params.ts_params);

  cudaFree(sentence_bn_g);
  cudaFree(sentence_bn_b);
  cudaFree(sentence_bn_ts);
  cudaFree(sentence_bn_tm);

  for (int i = 0; i < spec.num_combination_layers; i++)
    destroy_params(params.mlp_params[i]);

  cudaFree(classifier_W);
  cudaFree(classifier_b);
}


#endif

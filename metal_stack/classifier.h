#ifndef _classifier_
#define _classifier_

#include "cublas_v2.h"

#include "thin-stack.h"
#include "util.h"


typedef struct ClassifierParameters {
  float *embeddings;
  float *project_W;
  float *project_b;
  float *buffer_bn_ts;
  float *buffer_bn_tm;
  ThinStackParameters ts_params;
  float *sentence_bn_ts;
  float *sentence_bn_tm;
} ClassifierParameters;


class Classifier {
  public:

    Classifier(ModelSpec spec, ClassifierParameters params,
        cublasHandle_t handle);

    ~Classifier();

    ModelSpec spec;
    ClassifierParameters params;
    cublasHandle_t handle;
}


#endif

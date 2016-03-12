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
} ClassifierParameters;


#endif

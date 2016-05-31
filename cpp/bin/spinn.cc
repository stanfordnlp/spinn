#include "spinn.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"


void forward_batch()


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
}

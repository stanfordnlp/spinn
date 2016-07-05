#ifndef _util_
#define _util_

#include <assert.h>
#include <cstddef>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <curand.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

using namespace std;


#define DEBUG 0

class CUBLAS_HANDLE {
  public:
    static cublasHandle_t& getInstance() {
      static CUBLAS_HANDLE h;
      return h._handle;
    }

  private:
    CUBLAS_HANDLE() {
      cublasStatus_t stat = cublasCreate(&_handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "CUBLAS initialization failed (" << stat << ")" << endl;
      }
    }
    ~CUBLAS_HANDLE() { cublasDestroy(_handle); }

    cublasHandle_t _handle;
};

float *load_weights(string filename, int N);
float *load_weights(ifstream& file, int N);
float *load_weights_cuda(string filename, int N, float *target=NULL);
float *load_weights_cuda(ifstream& file, int N, float *target=NULL);

cublasHandle_t getCublasHandle();

void print_device_matrix(const float *m, int M, int N);
float *make_rand_matrix(int M, int N);
void fill_rand_matrix(float *m, int M, int N);

#endif

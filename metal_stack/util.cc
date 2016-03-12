#include "util.h"

float *load_weights(string filename, int N) {
  float *ret = (float *) malloc(N * sizeof(float));
  cout << filename << endl;
  ifstream file(filename);

  float x;
  for (int i = 0; i < N; i++) {
    file >> x;
    ret[i] = x;
  }

  return ret;
}

float *load_weights_cuda(string filename, int N) {
  float *h_weights = load_weights(filename, N);
  float *d_weights;
  cudaMalloc(&d_weights, N * sizeof(float));
  cudaMemcpy(d_weights, h_weights, N * sizeof(float),
      cudaMemcpyHostToDevice);
  free(h_weights);
  return d_weights;
}


cublasHandle_t getCublasHandle() {
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "CUBLAS initialization failed (" << stat << ")" << endl;
    return NULL;
  }
  return handle;
}


// Print a column-major matrix stored on device.
void print_device_matrix(const float *m, int M, int N) {
  cudaDeviceSynchronize();
  float *h_m = (float *) malloc(M * N * sizeof(float));
  cudaMemcpy(h_m, m, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cout << "[[ ";
  for (int i = 0; i < M; i++) {
    if (i > 0)
      cout << "   ";

    for (int j = 0; j < N; j++) {
      float val = h_m[j * M + i];
      printf(" %+.03f  ", val);
    }

    cout << " ]";
    if (i < M - 1)
      cout << endl;
  }
  cout << "]" << endl << endl;

  free(h_m);
}


void fill_rand_matrix(float *m, int M, int N) {
  static curandGenerator_t prng;
  if (!prng) {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  }

  curandGenerateUniform(prng, m, M * N);
}

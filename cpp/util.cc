#include "util.h"

float *move_var_to_device(float *h_source, int N, float *d_target=NULL) {
  if (!d_target)
    cudaMalloc(&d_target, N * sizeof(float));

  cudaMemcpy(d_target, h_source, N * sizeof(float), cudaMemcpyHostToDevice);
  free(h_source);
  return d_target;
}

float *load_weights(string filename, int N) {
  cout << filename << endl;
  ifstream file(filename);
  return load_weights(file, N);
}

float *load_weights(ifstream& file, int N) {
  float *ret = (float *) malloc(N * sizeof(float));

  float x;
  for (int i = 0; i < N; i++) {
    file >> x;
    ret[i] = x;
  }

  return ret;
}

float *load_weights_cuda(string file, int N, float *target) {
  float *h_weights = load_weights(file, N);
  return move_var_to_device(h_weights, N, target);
}

float *load_weights_cuda(ifstream& file, int N, float *target) {
  float *h_weights = load_weights(file, N);
  return move_var_to_device(h_weights, N, target);
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
      cout << " [ ";

    for (int j = 0; j < N; j++) {
      float val = h_m[j * M + i];
      printf(" %+.05f, ", val);
    }

    cout << " ],";
    if (i < M - 1)
      cout << endl;
  }
  cout << "]" << endl << endl;

  free(h_m);
}


float *make_rand_matrix(int M, int N) {
  float *m;
  cudaMalloc(&m, M * N * sizeof(float));
  fill_rand_matrix(m, M, N);
  return m;
}


void fill_rand_matrix(float *m, int M, int N) {
  static curandGenerator_t prng;
  if (!prng) {
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  }

  curandGenerateNormal(prng, m, M * N, 0.0f, 0.5f);
}

#include <iostream>
#include <ctime>
#include <math.h>
#include <ctime>
#include <stdio.h>
#include <cuda_runtime_api.h>


__device__ float sum(float a, float b) {
    return exp(a) + exp(b);
}


__global__ void add(long n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += step) {
        y[i] = sum(x[i], y[i]);
    }
}


int main() {

    long int N = 1 << 26;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 20M elements on the GPU
    time_t startTime = clock();
    add<<<1, 512>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    time_t duration = clock() - startTime;
    std::cout << duration << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    return 0;
}

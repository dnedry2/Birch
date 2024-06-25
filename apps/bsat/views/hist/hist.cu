#include "hist.cuh"
#include <cuda_runtime.h>

__global__ void calculateHistogram(const double* data, const bool* mask, double* hist, ulong dataLen, uint bins, double min, double binSize) {
    const ulong tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dataLen && mask[tid]) {
        const double val = data[tid];
        const uint index = (uint)((val - min) / binSize);
        atomicAdd(&hist[index], 1.0);
    }
}

double* buildHistCUDA(const double* data, const bool* mask, ulong dataLen, uint bins, double min, double max) {
    double* hist = new double[bins];
    // Initialize buffers to zero
    for (uint i = 0; i < bins; i++)
        hist[i] = 0;

    double* d_data;
    bool* d_mask;
    double* d_hist;

    const double binSize = (max - min) / bins;

    // Allocate device memory
    cudaMalloc((void**)&d_data, dataLen * sizeof(double));
    cudaMalloc((void**)&d_mask, dataLen * sizeof(bool));
    cudaMalloc((void**)&d_hist, bins * sizeof(double));

    // Copy input data to device memory
    cudaMemcpy(d_data, data, dataLen * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, dataLen * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, hist, bins * sizeof(double), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (dataLen + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate histogram on the GPU
    calculateHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_mask, d_hist, dataLen, bins, min, binSize);

    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(hist, d_hist, bins * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_mask);
    cudaFree(d_hist);

    return hist;
}
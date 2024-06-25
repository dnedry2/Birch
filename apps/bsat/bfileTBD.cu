#include <cuda_runtime.h>

__global__ void calcDTOAKernel(double* out, const double* in, const bool* mask, unsigned long elCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < elCount)
        if (mask == nullptr || mask[i])
            out[i] = in[i] - in[i - 1];
}

void calcDTOACUDA(double* out, const double* in, const bool* mask, unsigned long elCount) {
    cudaSetDevice(0);

    int threadsPerBlock = 256;
    int blocksPerGrid = (elCount + threadsPerBlock - 1) / threadsPerBlock;

    // Copy data to device
    double* d_in;
    double* d_out;
    bool*   d_mask;

    cudaMalloc(&d_in,   elCount * sizeof(double));
    cudaMalloc(&d_out,  elCount * sizeof(double));

    if (mask != nullptr)
        cudaMalloc(&d_mask, elCount * sizeof(bool));

    cudaMemcpy(d_in,   in,   elCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, elCount * sizeof(bool),   cudaMemcpyHostToDevice);
    
    calcDTOAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, mask == nullptr ? nullptr : d_mask, elCount);
    cudaDeviceSynchronize();

    // Copy data back to host
    cudaMemcpy(out, d_out, elCount * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    if (mask != nullptr)
        cudaFree(d_mask);
}
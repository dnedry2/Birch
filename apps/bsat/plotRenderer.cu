#include "cuda_runtime.h"

#include <stdio.h>

typedef unsigned int uint;

__global__ void renderPointsKernel(const double* xVals, const double* yVals, const bool* filt, const uint* col, unsigned long count, double xMin, double xMax, double yMin, double yMax, int xRes, int yRes, bool useFilter, bool* renderMask, uint* pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    double xScale = xRes / (xMax - xMin);
    double yScale = yRes / (yMax - yMin);

    double xVal = xVals[idx];
    double yVal = yVals[idx];

    if (yVal < yMin || yVal > yMax)
        return;
    if (xVal < xMin || xVal > xMax)
        return;
    if (useFilter && !filt[idx])
        return;

    unsigned int pos_x = (xVal - xMin) * xScale;
    unsigned int pos_y = (yVal - yMin) * yScale;
    unsigned int gridIdx = pos_y * xRes + pos_x;

    if (renderMask[gridIdx])
        return;

    renderMask[gridIdx] = true;
    pixels[gridIdx] = col[idx];
}

void renderPointsCUDA(const double* const xVals, const double* const yVals, const bool* const filt, const unsigned* const col, unsigned long count, double xMin, double xMax, double yMin, double yMax, int xRes, int yRes, bool useFilter, bool* renderMask, unsigned* pixels) {
    const int numThreads = 1024;
    const int numBlocks = (count + numThreads - 1) / numThreads;

    // Select GPU
    cudaSetDevice(0);

    // Copy data to device
    double* cxVals;
    double* cyVals;
    bool*   cfilt;
    uint*   ccol;
    bool*   cmask;
    uint*   cpix;

    cudaMalloc(&cxVals, count * sizeof(double));
    cudaMalloc(&cyVals, count * sizeof(double));
    cudaMalloc(&cfilt,  count * sizeof(bool));
    cudaMalloc(&ccol,   count * sizeof(uint));
    cudaMalloc(&cmask,  xRes  * yRes * sizeof(bool));
    cudaMalloc(&cpix,   xRes  * yRes * sizeof(uint));

    cudaMemcpy(cxVals, xVals,         count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cyVals, yVals,         count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cfilt,  filt,          count * sizeof(bool),   cudaMemcpyHostToDevice);
    cudaMemcpy(ccol,   col,           count * sizeof(uint),   cudaMemcpyHostToDevice);
    cudaMemcpy(cmask,  renderMask,    xRes  * yRes * sizeof(bool),  cudaMemcpyHostToDevice);
    cudaMemcpy(cpix,   pixels,        xRes  * yRes * sizeof(uint),  cudaMemcpyHostToDevice);

    renderPointsKernel<<<numBlocks, numThreads>>>(cxVals, cyVals, cfilt, ccol, count, xMin, xMax, yMin, yMax, xRes, yRes, useFilter, cmask, cpix);
    cudaDeviceSynchronize();

    // Copy data back to host
    cudaMemcpy(renderMask, cmask, xRes * yRes * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(pixels,     cpix,  xRes * yRes * sizeof(uint), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(cxVals);
    cudaFree(cyVals);
    cudaFree(cfilt);
    cudaFree(ccol);
    cudaFree(cmask);
    cudaFree(cpix);
}


__global__ void rotateCCKernel(const unsigned* data, unsigned* buf, int width, int height) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < width && tid_y < height) {
        buf[tid_x * height + tid_y] = data[tid_y * width + (width - 1) - tid_x];
    }
}

unsigned* rotateCCCUDA(const unsigned* data, int width, int height) {
    unsigned* buf = nullptr;
    unsigned* d_data = nullptr;
    unsigned* d_buf = nullptr;

    // Select GPU
    cudaSetDevice(0);

    // Allocate memory for the buffer on host and device
    try {
        buf = new unsigned[width * height];
        cudaMalloc((void**)&d_data, width * height * sizeof(unsigned));
        cudaMalloc((void**)&d_buf, width * height * sizeof(unsigned));
    } catch (...) {
        return nullptr;
    }

    // Copy input data from host to device
    cudaMemcpy(d_data, data, width * height * sizeof(unsigned), cudaMemcpyHostToDevice);

    // Define CUDA kernel launch configuration
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    rotateCCKernel<<<gridDim, blockDim>>>(d_data, d_buf, width, height);

    // Copy output data from device to host
    cudaMemcpy(buf, d_buf, width * height * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_buf);

    return buf;
}


__device__ uint blendPixels(uint dest, uint src) {
    // Perform blending operation here and return the result
    // This implementation assumes a simple alpha blending:
    unsigned char srcAlpha = (src >> 24) & 0xFF;
    unsigned char destAlpha = (dest >> 24) & 0xFF;
    unsigned char alpha = srcAlpha + destAlpha - (srcAlpha * destAlpha) / 255;
    unsigned char invAlpha = 255 - alpha;
    unsigned char red = (src & 0xFF) * srcAlpha / 255 + (dest & 0xFF) * invAlpha / 255;
    unsigned char green = ((src >> 8) & 0xFF) * srcAlpha / 255 + ((dest >> 8) & 0xFF) * invAlpha / 255;
    unsigned char blue = ((src >> 16) & 0xFF) * srcAlpha / 255 + ((dest >> 16) & 0xFF) * invAlpha / 255;
    return (alpha << 24) | (blue << 16) | (green << 8) | red;
}

__global__ void blitKernel(uint* dest, const uint* src, uint dWidth, uint dHeight, uint sWidth, uint sHeight, uint centerX, uint centerY, uint color) {
    int startX = centerX - sWidth / 2;
    int startY = centerY - sHeight / 2;

    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;

    if (sx >= 0 && sx < sWidth && sy >= 0 && sy < sHeight) {
        int dx = startX + sx;
        int dy = startY + (sHeight - 1 - sy);

        // Check if the destination coordinates are within the bounds of the destination buffer
        if (dx >= 0 && dx < dWidth && dy >= 0 && dy < dHeight) {
            uint srcCol = src[sy * sWidth + sx];
            uint destIdx = dy * dWidth + dx;

            // Extract the alpha channel from the source pixel
            unsigned char alpha = (srcCol >> 24) & 0xFF;

            if (alpha > 250) // If the source pixel is opaque, overwrite the destination pixel
                dest[destIdx] = (srcCol & 0xff000000) | (color & 0x00ffffff);
            else if (alpha == 0x00) // If the source pixel is transparent, skip
                return;
            else { // If the source pixel is semi-transparent, blend the source and destination pixels
                dest[destIdx] = blendPixels(dest[destIdx], srcCol);
            }
        }
    }
}

void blitCUDA(uint* dest, const uint* src, uint dWidth, uint dHeight, uint sWidth, uint sHeight, uint centerX, uint centerY, uint color) {
    cudaSetDevice(0);
    
    // Calculate grid and block dimensions for the CUDA kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((sWidth + blockDim.x - 1) / blockDim.x, (sHeight + blockDim.y - 1) / blockDim.y);

    // Call the CUDA kernel
    blitKernel<<<gridDim, blockDim>>>(dest, src, dWidth, dHeight, sWidth, sHeight, centerX, centerY, color);
}
unsigned* copyToCUDA(const unsigned* data, unsigned length) {
    unsigned* data_cu = nullptr;

    // Select GPU
    cudaSetDevice(0);

    // Allocate memory on device
    cudaMalloc((void**)&data_cu, length * sizeof(unsigned));

    // Copy data to device
    cudaMemcpy(data_cu, data, length * sizeof(unsigned), cudaMemcpyHostToDevice);

    return data_cu;
}
void freeCUDA(void* data) {
    cudaFree(data);
}
void copyFromCUDA(unsigned* dest, const unsigned* src_cu, unsigned length) {
    // Copy data from device
    cudaMemcpy(dest, src_cu, length * sizeof(unsigned), cudaMemcpyDeviceToHost);
}

void syncCUDA() {
    cudaSetDevice(0);
    cudaDeviceSynchronize();
}
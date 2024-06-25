#include <cstdio>
#include "cufft.h"

#include "cudaFuncs.cuh"
#include "logger.hpp"

int cuda_count_devices() {
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);

    return nDevices;
}

void cuda_list_devices() {
    int nDevices = cuda_count_devices();

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        DispInfo("Birch", "Compute device %d: %s", i + 1, prop.name);
    }
}

void cuda_init() {
    cufftHandle handle;
    cufftPlan1d(&handle, 2048, cufftType_t::CUFFT_Z2Z, 1);

    cufftDestroy(handle);
}

void cuda_get_device_memory(int gpu, size_t *free, size_t *total) {
    cudaSetDevice(gpu);
    cudaMemGetInfo(free, total);
}
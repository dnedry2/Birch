#include "fft.h"

#include <cstdio>
#include "cufft.h"

using namespace Birch;

struct cudaPlan {
    cufftHandle handle;

    cufftDoubleComplex* inBuf  = nullptr;
    cufftDoubleComplex* outBuf = nullptr;

    const unsigned size = 0;
    const unsigned cnt  = 0;
    const unsigned bufferBytes = 0;
    const unsigned device = 0;

    cudaPlan(unsigned size, unsigned cnt, unsigned dev) : size(size), cnt(cnt), bufferBytes(size * cnt * sizeof(*outBuf)), device(dev) { }
};

fft_gpu_plan fft_cuda_make_plan(unsigned maxSize, unsigned fftCnt, unsigned dev) {
    cudaSetDevice(dev);

    cudaPlan* plan = new cudaPlan(maxSize, fftCnt, dev);

    cufftPlan1d(&plan->handle, maxSize, cufftType_t::CUFFT_Z2Z, fftCnt);
    cudaMalloc((void**)&plan->inBuf,  plan->bufferBytes);
    cudaMalloc((void**)&plan->outBuf, plan->bufferBytes);

    return static_cast<fft_gpu_plan>(plan);
}
void fft_cuda_destroy_plan(fft_gpu_plan plan) {
    cudaPlan* const cPlan = static_cast<cudaPlan*>(plan);

    cudaSetDevice(cPlan->device);

    cufftDestroy(cPlan->handle);
    cudaFree(cPlan->inBuf);
    cudaFree(cPlan->outBuf);

    delete cPlan;
}

void fft_cuda_cpx_forward(fft_gpu_plan plan, Complex<double>* input, Complex<double>* output) {
    cudaPlan* const cPlan = static_cast<cudaPlan*>(plan);
    cudaSetDevice(cPlan->device);

    cudaMemcpy((void*)cPlan->inBuf, (void*)input, cPlan->bufferBytes, cudaMemcpyHostToDevice);

    cufftExecZ2Z(cPlan->handle, cPlan->inBuf, cPlan->outBuf, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    cudaMemcpy((void*)output, (void*)cPlan->outBuf, cPlan->bufferBytes, cudaMemcpyDeviceToHost);
}
void fft_cuda_cpx_inverse(fft_gpu_plan plan, Complex<double>* input, Complex<double>* output) {
    cudaPlan* const cPlan = static_cast<cudaPlan*>(plan);
    cudaSetDevice(cPlan->device);

    cudaMemcpy((void*)cPlan->inBuf, (void*)input, cPlan->bufferBytes, cudaMemcpyHostToDevice);

    cufftExecZ2Z(cPlan->handle, cPlan->inBuf, cPlan->outBuf, CUFFT_INVERSE);
    cudaDeviceSynchronize();

    cudaMemcpy((void*)output, (void*)cPlan->outBuf, cPlan->bufferBytes, cudaMemcpyDeviceToHost);
}
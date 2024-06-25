#include "fft.h"
#include "fft_impl_cuda.cuh"

using namespace Birch;

fft_gpu_plan fft_gpu_make_plan(unsigned fftSize, unsigned blockSize, unsigned device) { return fft_cuda_make_plan(fftSize, blockSize, device); }
void fft_gpu_destroy_plan(fft_plan plan) { return fft_cuda_destroy_plan(plan); }

void fft_gpu_cpx_inverse(fft_plan plan, Complex<double>* input, Complex<double>* output) { fft_cuda_cpx_inverse(plan, input, output); }
void fft_gpu_cpx_forward(fft_plan plan, Complex<double>* input, Complex<double>* output) { fft_cuda_cpx_forward(plan, input, output); }
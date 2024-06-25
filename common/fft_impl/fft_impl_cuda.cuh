#include "fft.h"

fft_gpu_plan fft_cuda_make_plan(unsigned maxSize, unsigned fftCnt, unsigned device);
void fft_cuda_destroy_plan(fft_plan plan);
void fft_cuda_cpx_forward(fft_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);
void fft_cuda_cpx_inverse(fft_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);
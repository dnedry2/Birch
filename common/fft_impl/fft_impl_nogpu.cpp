#include "fft.h"
#include "logger.hpp"

using namespace Birch;

fft_gpu_plan fft_gpu_make_plan(unsigned size) {
    DispError("fft_gpu_make_plan", "Placeholder code - should not be reached...");
    return nullptr;
}
void fft_gpu_destroy_plan(fft_plan plan) { DispError("fft_gpu_destroy_plan", "Placeholder code - should not be reached..."); }

void fft_gpu_cpx_inverse(fft_plan plan, Complex<double>* input, Complex<double>* output) { DispError("fft_gpu_cpx_inverse", "Placeholder code - should not be reached..."); }
void fft_gpu_cpx_forward(fft_plan plan, Complex<double>* input, Complex<double>* output) { DispError("fft_gpu_cpx_forward", "Placeholder code - should not be reached..."); }
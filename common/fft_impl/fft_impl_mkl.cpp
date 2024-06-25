#include "fft.h"

#include <cmath>
#include <cstdio>

#include "mkl_dfti.h"

using namespace Birch;

fft_plan fft_make_plan(unsigned maxSize) {
    // FFTSetupD is just a typedef'd pointer
    DFTI_DESCRIPTOR_HANDLE* hand = (DFTI_DESCRIPTOR_HANDLE*)malloc(sizeof(DFTI_DESCRIPTOR_HANDLE));

    DftiCreateDescriptor(hand, DFTI_DOUBLE, DFTI_COMPLEX, 1, maxSize);
    DftiSetValue(*hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(*hand, DFTI_BACKWARD_SCALE, 1.0 / maxSize);
    DftiCommitDescriptor(*hand);

    return static_cast<fft_plan>(hand);
}
void fft_destroy_plan(fft_plan plan) {
    DftiFreeDescriptor(static_cast<DFTI_DESCRIPTOR_HANDLE*>(plan));
    free(plan);
}

void fft_cpx_forward(fft_plan plan, Complex<double>* input, Complex<double>* output) {
    DftiComputeForward(*static_cast<DFTI_DESCRIPTOR_HANDLE*>(plan), input, output);
}
void fft_cpx_inverse(fft_plan plan, Complex<double>* input, Complex<double>* output) {
    DftiComputeBackward(*static_cast<DFTI_DESCRIPTOR_HANDLE*>(plan), input, output);
}
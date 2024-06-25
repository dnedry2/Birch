#include "fft.h"

#include <cmath>
#include <cstdio>
#include <Accelerate/Accelerate.h>

#define FFT_IMPL_ACCELERATE

typedef struct {
    void* plan;
    int   size2n;
} fft_plan_;

unsigned log2n(unsigned in) {
    return static_cast<unsigned>(log2(in));
}

fft_plan fft_make_plan(unsigned size) {
    fft_plan_* plan = new fft_plan_();
    plan->plan = (void*)vDSP_create_fftsetupD(log2n(size), 0); // FFTSetupD is just a typedef'd pointer
    plan->size2n = log2n(size);

    return static_cast<fft_plan>(plan);
}
void fft_destroy_plan(fft_plan plan) {
    fft_plan_* plan_ = (fft_plan_*)plan;

    vDSP_destroy_fftsetupD(static_cast<FFTSetupD>(plan_->plan));
    delete plan_;
}

void fft_cpx_forward(fft_plan plan, Birch::Complex<double>* in, Birch::Complex<double>* out) {
    DSPDoubleSplitComplex input  = { &in[0].R,  &in[0].I };
    DSPDoubleSplitComplex output = { &out[0].R, &out[0].I };

    fft_plan_* plan_ = (fft_plan_*)plan;
    vDSP_fft_zopD(static_cast<FFTSetupD>(plan_->plan), &input, 2, &output, 2, plan_->size2n, kFFTDirection_Forward);
}   
void fft_cpx_inverse(fft_plan plan, Birch::Complex<double>* in, Birch::Complex<double>* out) {
    DSPDoubleSplitComplex input  = { &in[0].R,  &in[0].I };
    DSPDoubleSplitComplex output = { &out[0].R, &out[0].I };

    fft_plan_* plan_ = (fft_plan_*)plan;
    vDSP_fft_zopD(static_cast<FFTSetupD>(plan_->plan), &input, 2, &output, 2, plan_->size2n, kFFTDirection_Inverse);
}
void fft_r_forward(fft_plan plan, double* inR, double* outR) {

}
void fft_r_reverse(fft_plan plan, double* inR, double* outR) {

}
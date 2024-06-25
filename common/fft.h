#ifndef __FFT_H__
#define __FFT_H__

#include "include/birch.h"

typedef void* fft_plan;
typedef void* fft_gpu_plan;

const char* fft_get_impl();

fft_plan fft_make_plan(unsigned size);
void fft_destroy_plan(fft_plan plan);

void fft_cpx_inverse(fft_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);
void fft_cpx_forward(fft_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);

void fft_r_forward(fft_plan plan, double* inR, double* outR, unsigned size);
void fft_r_inverse(fft_plan plan, double* inR, double* outR, unsigned size);



fft_gpu_plan fft_gpu_make_plan(unsigned fftSize, unsigned blockSize, unsigned device);
void fft_gpu_destroy_plan(fft_gpu_plan plan);

void fft_gpu_cpx_inverse(fft_gpu_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);
void fft_gpu_cpx_forward(fft_gpu_plan plan, Birch::Complex<double>* input, Birch::Complex<double>* output);

void fft(Birch::Complex<double>* input, Birch::Complex<double>* output, unsigned size);
void ifft(Birch::Complex<double>* input, Birch::Complex<double>* output, unsigned size);

void fft_cleanup();

template <typename T>
void fftshift(Birch::Complex<T> *data, int count);
template <typename T>
void ifftshift(Birch::Complex<T> *data, int count);


void convolve(Birch::Complex<double>* sig, unsigned sigLen, Birch::Complex<double>* kernel, unsigned kernelLen, Birch::Complex<double>* output);



// Tests
#ifdef DEBUG
void _TEST_convolve();
#endif

#endif
#include <cstring>
#include <map>
#include "fft.h"

static const char impNameAccel[]   = "Apple Accelerate";
static const char impNameMKL[]     = "Intel MKL";
static const char impNameCuda[]    = "Nvidia cuFFT";
static const char impNameMKLCuda[] = "Intel MKL, Nvidia cuFFT";

#ifdef __APPLE__
    #include "fft_impl/fft_impl_accel.cpp"
#endif

#ifdef FFT_IMPL_CUDA
    #include "fft_impl/fft_impl_cuda.cpp"
#else
    #include "fft_impl/fft_impl_nogpu.cpp"
#endif
#if defined(_WIN64) || defined(_WIN32) || defined(__unix__)
    #include "fft_impl/fft_impl_mkl.cpp"
    #define FFT_IMPL_MKL
#endif

const char* fft_get_impl() {


#ifdef FFT_IMPL_FFTW3
    return "FFTW3";
#endif
#ifdef FFT_IMPL_ACCELERATE
    return impNameAccel;
#endif
#if defined(FFT_IMPL_MKL) && defined(FFT_IMPL_CUDA)
    return impNameMKLCuda;
#endif
#ifdef FFT_IMPL_MKL
    return impNameMKL;
#endif
#ifdef FFT_IMPL_CUDA
    return impNameCuda;
#endif
}

template <typename T>
static inline void swap(Birch::Complex<T> *v1, Birch::Complex<T> *v2)
{
    Birch::Complex<T> tmp = *v1;
    *v1 = *v2;
    *v2 = tmp;
}

template <typename T>
void fftshift(Birch::Complex<T> *data, int count)
{
    int k = 0;
    int c = (int) floor((float)count/2);
    // For odd and for even numbers of element use different algorithm
    if (count % 2 == 0)
    {
        for (k = 0; k < c; k++)
            swap(&data[k], &data[k+c]);
    }
    else
    {
        Birch::Complex<T> tmp = data[0];
        for (k = 0; k < c; k++)
        {
            data[k] = data[c + k + 1];
            data[c + k + 1] = data[k + 1];
        }
        data[c] = tmp;
    }
}

template <typename T>
void ifftshift(Birch::Complex<T> *data, int count)
{
    int k = 0;
    int c = (int) floor((float)count/2);
    if (count % 2 == 0)
    {
        for (k = 0; k < c; k++)
            swap(&data[k], &data[k+c]);
    }
    else
    {
        Birch::Complex<T> tmp = data[count - 1];
        for (k = c-1; k >= 0; k--)
        {
            data[c + k + 1] = data[k];
            data[k] = data[c + k];
        }
        data[c] = tmp;
    }
}

template void fftshift<float>(Birch::Complex<float> *data, int count);
template void fftshift<double>(Birch::Complex<double> *data, int count);

template void ifftshift<float>(Birch::Complex<float> *data, int count);
template void ifftshift<double>(Birch::Complex<double> *data, int count);

static std::map<unsigned, fft_plan> _fftPlans;

void fft(Birch::Complex<double>* input, Birch::Complex<double>* output, unsigned size)
{
    _fftPlans.emplace(size, fft_make_plan(size));
    fft_cpx_forward(_fftPlans[size], input, output);
}
void ifft(Birch::Complex<double>* input, Birch::Complex<double>* output, unsigned size)
{
    _fftPlans.emplace(size, fft_make_plan(size));
    fft_cpx_inverse(_fftPlans[size], input, output);
}
void fftCleanup()
{
    for (const auto &plan : _fftPlans)
        fft_destroy_plan(plan.second);

    _fftPlans.clear();
}

#ifdef DEBUG
template <typename T>
static bool fEq(const T& a, const T& b, double tol = 1e-12)
{
    return fabs(a - b) < tol;
};

// Test the convolution function
void _TEST_convolve() {
    // Test case 1: Convolution of two sine waves

    const uint sigLen = 64;
    const uint kernelLen = 16;
    const uint outputLen = sigLen + kernelLen - 1;

    Birch::Complex<double> sig[sigLen];
    Birch::Complex<double> kernel[kernelLen];
    Birch::Complex<double> output[outputLen];

    // Generate input signals
    for (uint i = 0; i < sigLen; i++)
        sig[i] = sin(2 * M_PI * i / sigLen);

    for (uint i = 0; i < kernelLen; i++)
        kernel[i] = sin(2 * M_PI * i / kernelLen);

    // Call the function to compute the convolution
    convolve(sig, sigLen, kernel, kernelLen, output);

    for (uint i = 0; i < sigLen; i++)
        printf("%f + %fi\n", output[i].R, output[i].I);

    // Verify the result
    /*
    for (uint i = 0; i < outputLen; i++) {
        Birch::Complex<double> expected(0, 0);
        for (uint j = 0; j < kernelLen; j++) {
            if (i >= j && i < j + sigLen)
                expected += kernel[j] * sig[i - j];
        }

        if (!fEq(expected.R, output[i].R) || !fEq(expected.I, output[i].I))
            printf("Error at %d: %f + %fi != %f + %fi\n", i, expected.R, expected.I, output[i].R, output[i].I);
    }
    */
}
#endif
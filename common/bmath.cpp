#include "fft.h"

#include <cmath>
#include <algorithm>

using namespace Birch;

// Filter using overlap-add method
void firFilter(Complex<double>* input, uint size, Complex<double>* impulse, uint impulseSize) {
    if (size <= impulseSize)
         static_assert(true, "Input size must be greater than impulse size");

    uint fftSize = 1;
    while (pow(2, ++fftSize) < impulseSize * 4);
    fftSize = pow(2, fftSize);

    fft_plan plan = fft_make_plan(fftSize);

    Complex<double>* paddedImpulse = new Complex<double>[fftSize];
    std::fill_n(paddedImpulse, fftSize, Complex<double>(0, 0));
    std::copy(impulse, impulse + impulseSize, paddedImpulse);

    Complex<double>* impulseFreq = new Complex<double>[fftSize];
    Complex<double>* signalFreq  = new Complex<double>[fftSize];
    Complex<double>* convolved   = new Complex<double>[fftSize];
    Complex<double>* result      = new Complex<double>[fftSize];

    Complex<double>* output = new Complex<double>[size + fftSize - 1];

    fft_cpx_forward(plan, paddedImpulse, impulseFreq);

    const uint chunkSize = fftSize - impulseSize + 1;

    for (uint i = 0; i < size; i += chunkSize) {
        // Pad the input with zeros if necessary
        if (i + chunkSize > size) {
            Birch::Complex<double>* paddedInput = new Birch::Complex<double>[fftSize];

            std::fill_n(paddedInput, fftSize, Birch::Complex<double>(0, 0));
            std::copy(input + i, input + size - i * chunkSize, paddedInput);

            fft_cpx_forward(plan, paddedInput, signalFreq);

            delete[] paddedInput;
        } else {
            fft_cpx_forward(plan, input + i, signalFreq);
        }

        for (uint j = 0; j < fftSize; ++j)
            convolved[j] = signalFreq[j] * impulseFreq[j];

        fft_cpx_inverse(plan, convolved, result);

        for (uint j = 0; j < chunkSize; ++j)
            output[i + j] = output[i + j] + result[j];
    }


    std::copy(output, output + size, input);


    // Cleanup
    fft_destroy_plan(plan);

    delete[] paddedImpulse;
    delete[] impulseFreq;
    delete[] signalFreq;
    delete[] convolved;
    delete[] result;
    delete[] output;
}

Complex<double>* createLowPassFilter(unsigned impulseSize, double ratio) {
    Complex<double>* filter = new Complex<double>[impulseSize];

    double cutoffFrequency = 0.5 / ratio;

    for (unsigned n = 0; n < impulseSize; ++n) {
        double t = (n - static_cast<double>(impulseSize - 1) / 2.0) / ratio;

        double sinc   = (t == 0.0) ? 1.0 : (sin(2.0 * M_PI * cutoffFrequency * t) / (M_PI * t));
        double window = 0.54 - 0.46 * cos(2.0 * M_PI * n / (impulseSize - 1));

        filter[n].R = sinc * window;
        filter[n].I = 0.0;
    }

    return filter;
}

// Resample via lerp + low-pass
Birch::Complex<double>* resample(Birch::Complex<double>* input, unsigned size, double inSampleRate, double outSampleRate, unsigned* outSize) {
    double ratio = outSampleRate / inSampleRate;

    *outSize = static_cast<unsigned>(size * ratio);
    Complex<double>* output = new Complex<double>[*outSize];

    for (unsigned i = 0; i < *outSize; ++i) {
        double   originalIndex = i / ratio;
        unsigned indexFloor = static_cast<unsigned>(std::floor(originalIndex));
        unsigned indexCeil = std::min(indexFloor + 1, size - 1);

        double frac = originalIndex - indexFloor;

        // Linear interpolation
        output[i].R = (1 - frac) * input[indexFloor].R + frac * input[indexCeil].R;
        output[i].I = (1 - frac) * input[indexFloor].I + frac * input[indexCeil].I;
    }

/*
    if (*outSize > 1024) {
        Complex<double>* filterImpulseResponse = createLowPassFilter(1024, ratio);
        firFilter(output, *outSize, filterImpulseResponse, 1024);
        delete[] filterImpulseResponse;
    }
*/
    return output;
}
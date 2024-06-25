#include <cmath>

#include "fir.hpp"
#include "fft.h"
#include "window.hpp"
#include "windowImpls.hpp"

using namespace Birch;

static unsigned calcFFTSize(unsigned taps) {
    unsigned out = 1;
    while (pow(2, ++out) < taps * 4);

    return out;
}
static unsigned calcFFTSizeClose(unsigned taps) {
    unsigned out = 1;
    while (pow(2, ++out) < taps);

    return out;
}

BandpassFIR::BandpassFIR(double bw, unsigned taps) : bandwidth(bw), taps(taps) { }

void BandpassFIR::SetBandpass(double center, double bw, unsigned size, WindowFunc* window, bool stop) {
    // Design desired freq response
    float* fd = new float[size];

    double start = center - (bw / 2);
    double end   = center + (bw / 2);

    if (start < -bandwidth / 2)
        start = -bandwidth / 2;
    if (end > bandwidth / 2)
        end = bandwidth / 2;

    if (start > end) {
        auto t = end;

        end = start;
        start = t;
    }

    const float scale = 1 / (bandwidth / taps);

    unsigned startEl = start * scale + size / 2;
    unsigned endEl   = end   * scale + size / 2;

    if (!stop) {
        for (unsigned i = 0; i < taps; i++)
            fd[i] = 0;
        for (unsigned i = startEl; i < endEl; i++)
            fd[i] = 1;
    } else {
        for (unsigned i = 0; i < taps; i++)
            fd[i] = 1;
        for (unsigned i = startEl; i < endEl; i++)
            fd[i] = 0;
    }

    SetFilter(fd, size, window);

    delete[] fd;
}
void BandpassFIR::SetFilter(const float* design, unsigned taps, WindowFunc* window) {
    // Reset buffers
    this->taps = taps;
    this->fftSize = pow(2, calcFFTSize(taps));

    delete[] freqResp;
    delete[] impulseRes;
    delete[] this->design;

    freqResp     = new Birch::Complex<double>[fftSize];
    impulseRes   = new Birch::Complex<double>[taps];
    this->design = new Birch::Complex<double>[taps];

    for (unsigned i = 0; i < taps; i++) {
        this->design[i].R = design[i];
        this->design[i].I = 0;
    }

    // Find impulse response
    auto plan = fft_make_plan(taps);

    fft_cpx_inverse(plan, this->design, impulseRes);
    
    ifftshift(impulseRes, taps);

    auto win = window->Build(taps);

    for (unsigned i = 0; i < taps; i++) {
        impulseRes[i].R *= win[i];
        impulseRes[i].I *= win[i];
    }

    delete[] win;
    fft_destroy_plan(plan);
}
const Birch::Complex<double>* BandpassFIR::ImpulseResponse() {
    return impulseRes;
}
const Birch::Complex<double>* BandpassFIR::Design() {
    return design;
}
unsigned BandpassFIR::FFTSize() {
    return fftSize;
}
unsigned BandpassFIR::TapCount() {
    return taps;
}
BandpassFIR::~BandpassFIR() {
    delete[] freqResp;
    delete[] impulseRes;
    delete[] design;
}
bool BandpassFIR::Shift() {
    return true;
}
bool BandpassFIR::Peaks() {
    return false;
}

void BandpassFIR::Apply(Birch::Complex<double>* input) {
    const Birch::Complex<double>* const kernel = freqResp;
    const int len = taps;

    for (int i = 0; i < len; i++) {
        const Birch::Complex<double> in = input[i];
        const Birch::Complex<double> k  = kernel[i];

        input[i].R = in.R * k.R - in.I * k.I;
        input[i].I = (in.R + in.I) * (k.R + k.I) - input[i].R;
    }
}



MatchFIR::MatchFIR(const double* kR, const double* kI, unsigned size) {
    /*

    taps    = 2048;
    fftSize = 4096;
    impResp = new Birch::Complex<double>[taps];

    uint     chunks = std::ceil(size / (double)taps);
    auto     impl   = new Birch::Complex<double>[taps];
    auto     freq   = new Birch::Complex<double>[taps];
    auto     kernel = new Birch::Complex<double>[taps];
    fft_plan plan   = fft_make_plan(taps);


    std::fill_n(kernel, taps, Birch::Complex<double>(0, 0));

    double kMax = 0;

    for (uint i = 0; i < size; i++) {
        if (fabs(kR[i]) > kMax)
            kMax = fabs(kR[i]);
        if (fabs(kI[i]) > kMax)
            kMax = fabs(kI[i]);
    }

    for (uint i = 0; i < chunks; i++) {
        uint start = i * taps;

        // Normalize chunk
        for (uint j = 0; j < taps; j++) {
            if (start + j < size) {
                impl[j].R = kI[start + j] / chunks;
                impl[j].I = kR[start + j] / chunks;
            } else {
                impl[j] = 0;
            }
        }


        fft_cpx_forward(plan, impl, freq);


        for (uint j = 0; j < taps; j++)
            kernel[j] += freq[j];
    }

    fft_cpx_inverse(plan, kernel, impResp);

    // Normalize
    double max = 0;

    for (uint i = 0; i < taps; i++) {
        if (fabs(impResp[i].R) > max)
            max = fabs(impResp[i].R);
        if (fabs(impResp[i].I) > max)
            max = fabs(impResp[i].I);
    }

    for (uint i = 0; i < taps; i++) {
        impResp[i].R /= max;
        impResp[i].I /= max;
    }

    // Reverse
    for (uint i = 0; i < taps / 2; i++) {
        Birch::Complex<double> t = impResp[i];
        impResp[i] = impResp[taps - i - 1];
        impResp[taps - i - 1] = t;
    }

    fft_destroy_plan(plan);
    delete[] impl;
    delete[] freq;
    delete[] kernel;
    */

    fftSize = pow(2, calcFFTSizeClose(size));

    impResp = new Birch::Complex<double>[fftSize];
    
    Birch::Complex<double> zero = { 0, 0 };
    for (unsigned i = 0; i < fftSize; i++)
        impResp[i] = zero;

    double max = 0;

    impResp[0].R = kI[0];
    impResp[0].I = kR[0];

    for (unsigned i = 0; i < size - 1; i++) {
        impResp[size - i].R = kI[i];
        impResp[size - i].I = kR[i];

        if (fabs(impResp[i].R) > max)
            max = fabs(impResp[i].R);
        if (fabs(impResp[i].I) > max)
            max = fabs(impResp[i].I);
    }

    for (unsigned i = 0; i < fftSize; i++) {
        impResp[i].R /= max;
        impResp[i].I /= max;
    }

    /*{
        auto f = fopen("match.dat", "w");

        for (int i = 0; i < taps; i++) {
            Birch::Complex<float> out = { (float)impResp[i].R, (float)impResp[i].I };
            fwrite(&out, sizeof(out), 1, f);
        }
        for (int i = taps; i < fftSize; i++) {
            Birch::Complex<float> out = { 0, 0 };
            fwrite(&out, sizeof(out), 1, f);
        }

        fclose(f);
    }*/
}

MatchFIR::~MatchFIR() {
    delete[] impResp;
}

unsigned MatchFIR::FFTSize() {
    return fftSize;
}
unsigned MatchFIR::TapCount() {
    return taps;
}

const Birch::Complex<double>* MatchFIR::ImpulseResponse() {
    return impResp;
}

bool MatchFIR::Shift() {
    return false;
}
bool MatchFIR::Peaks() {
    return false;
}


SmoothFIR::SmoothFIR(unsigned taps) : taps(taps) {
    fftSize    = pow(2, calcFFTSize(taps));
    impulseRes = new Birch::Complex<double>[taps];

    std::fill_n(impulseRes, taps, Birch::Complex<double>(1.0 / taps, 1.0 / taps));

    // write to file

    {
        auto f = fopen("smooth.dat", "w");

        for (int i = 0; i < taps; i++) {
            Birch::Complex<float> out = { (float)impulseRes[i].R, (float)impulseRes[i].I };
            fwrite(&out, sizeof(out), 1, f);
        }
        for (int i = taps; i < fftSize; i++) {
            Birch::Complex<float> out = { 0, 0 };
            fwrite(&out, sizeof(out), 1, f);
        }

        fclose(f);
    }
}
SmoothFIR::~SmoothFIR() {
    delete[] impulseRes;
}

unsigned SmoothFIR::FFTSize() {
    return fftSize;
}
unsigned SmoothFIR::TapCount() {
    return taps;
}
bool SmoothFIR::Shift() {
    return false;
}
bool SmoothFIR::Peaks() {
    return false;
}

const Birch::Complex<double>* SmoothFIR::ImpulseResponse() {
    return impulseRes;
}
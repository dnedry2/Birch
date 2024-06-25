#ifndef __FIR_H__
#define __FIR_H__

#include "fft.h"
#include "window.hpp"

class FIR {
public:
    virtual unsigned TapCount() = 0;
    virtual unsigned FFTSize() = 0;
    virtual bool Shift() = 0;
    virtual bool Peaks() = 0; // Should reader find peaks?

    virtual const Birch::Complex<double>* ImpulseResponse() = 0;

    virtual ~FIR() { }
};

class BandpassFIR : public FIR {
public:
    void Apply(Birch::Complex<double>* input);
    void SetBandpass(double center, double bw, unsigned taps, WindowFunc* window, bool stop);
    void SetFilter(const float* design, unsigned taps, WindowFunc* window);
    unsigned TapCount() override;
    bool Shift() override;
    bool Peaks() override;

    const Birch::Complex<double>* ImpulseResponse() override;
    unsigned FFTSize() override;
    const Birch::Complex<double>* Design();

    BandpassFIR(double bw, unsigned taps);
    ~BandpassFIR();

private:
    Birch::Complex<double>* design     = nullptr;
    Birch::Complex<double>* freqResp   = nullptr;
    Birch::Complex<double>* impulseRes = nullptr;
    unsigned taps;
    unsigned fftSize;
    const double bandwidth;
};

class MatchFIR  : public FIR {
public:
    unsigned TapCount() override;
    bool Shift() override;
    bool Peaks() override;

    const Birch::Complex<double>* ImpulseResponse() override;
    unsigned FFTSize() override;

    MatchFIR(const double* kR, const double* kI, unsigned size);
    ~MatchFIR();

private:
    Birch::Complex<double>* impResp = nullptr;
    unsigned taps;
    unsigned fftSize;
};

class SmoothFIR : public FIR {
public:
    unsigned TapCount() override;
    unsigned FFTSize() override;
    bool Shift() override;
    bool Peaks() override;

    const Birch::Complex<double>* ImpulseResponse() override;

    SmoothFIR(unsigned taps);
    ~SmoothFIR();

private:
    Birch::Complex<double>* impulseRes = nullptr;
    unsigned taps;
    unsigned fftSize;
};

#endif
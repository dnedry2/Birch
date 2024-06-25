#ifndef __CDIF__
#define __CDIF__

#include <vector>

#include "include/birch.h"
#include "fir.hpp"
#include "window.hpp"

class SignalCDIF {
public:
    unsigned long* ElementCount(); // TODO: This should be const. Will fix after I update plotfield
    unsigned long Size();
    double SampleInterval();
    double SampleRate();
    double Bandwidth();
    unsigned& Stride();
    Birch::Timespan& Time();

    double* TOA();
    double* Amplitude();
    double* Phase();
    double* PhaseWrapped();
    double* Freq();
    double* RI();
    double* RQ();
    double* SI();
    double* SQ();
    double*& Spectra();
    double*& SpectraTotal();
    std::vector<double>* PeaksTOA();
    std::vector<double>* PeaksAmp();

    unsigned long* SpectraCount();
    unsigned& SpectraXRes();
    unsigned& SpectraYRes();
    Birch::Timespan& SpectraXOffset();
    Birch::Timespan& SpectraYOffset();
    Birch::Timespan& SpectraFreqSpan();

    SignalCDIF(unsigned long size, double sampleInterval);
    ~SignalCDIF();

    SignalCDIF(const SignalCDIF&) = delete;
    SignalCDIF& operator=(const SignalCDIF&) = delete;
private:
    unsigned long elCount = 0;
    unsigned long size    = 0;
    double sampleInterval = 0;
    unsigned stride       = 0;
    Birch::Timespan span;

    double* toa     = nullptr;
    double* amp     = nullptr;
    double* phase   = nullptr;
    double* phaseW  = nullptr;
    double* freq    = nullptr;
    double* ri      = nullptr;
    double* rq      = nullptr;
    double* si      = nullptr;
    double* sq      = nullptr;
    double* spectra = nullptr;
    double* spectraTotal = nullptr;

    unsigned long spectraCount = 0;
    unsigned      spectraXRes  = 0;
    unsigned      spectraYRes  = 0;
    Birch::Timespan      spectraXOffset;
    Birch::Timespan      spectraYOffset;
    Birch::Timespan      spectraFreqSpan;

    std::vector<double> peaksToa;
    std::vector<double> peaksAmp;
};
// Tune
// z'(t) = z(t) · e ^ ( -i · 2 · PI · f_shift · t)
class CDIFReader {
friend SignalCDIF;

public:
    void Process(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, unsigned stride, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned specRes, volatile float* progress, volatile bool* stop);
    void Spectra(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, Birch::Span<double> freqSpan, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned res, volatile float *progress, volatile bool *stop);

    CDIFReader(unsigned cacheSize);
    ~CDIFReader();

protected:
    // Should only work on one thread at a time
    bool     working         = false;
    unsigned cacheTotalSize  = 1073741824;
    unsigned threadCount     = 0;
    unsigned threadCacheSize = 0;
    unsigned cacheCount      = 0;
    char*    memory          = nullptr;

    template<typename T>
    void process(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, unsigned stride, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned specRes, volatile float* progress, volatile bool* stop);
    template<typename T>
    void spectra(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, Birch::Span<double> freqSpan, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned res, volatile float *progress, volatile bool *stop);
};

#endif
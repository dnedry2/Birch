#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <thread>
#include <vector>
#include <algorithm>

#include "cdif.hpp"

#include "fft.h"
#include "cudaFuncs.cuh"

#include "stopwatch.hpp"
#include "logger.hpp"
//#include "endian.h"
#include "fir.hpp"

#ifdef DEBUG
#include <iostream>
using std::cout;
using std::endl;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#endif

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Birch;
using std::vector;
using std::thread;

SignalCDIF::SignalCDIF(unsigned long size, double sampleInterval) {
    this->size = size;
    this->sampleInterval = sampleInterval;

    toa     = new double[size];
    amp     = new double[size];
    phase   = new double[size];
    phaseW  = new double[size];
    freq    = new double[size];
    ri      = new double[size];
    rq      = new double[size];
    si      = new double[size];
    sq      = new double[size];
    spectra = new double[size];
}
SignalCDIF::~SignalCDIF() {
    delete[] toa;
    delete[] amp;
    delete[] phase;
    delete[] phaseW;
    delete[] freq;
    delete[] ri;
    delete[] rq;
    delete[] si;
    delete[] sq;
    delete[] spectra;
}
unsigned long* SignalCDIF::ElementCount() { return &elCount; }
unsigned long SignalCDIF::Size() { return size; }
double SignalCDIF::SampleInterval() { return sampleInterval; }
double SignalCDIF::SampleRate() { return 1.0 / sampleInterval; }
unsigned& SignalCDIF::Stride() { return stride; }
Birch::Timespan& SignalCDIF::Time() { return span; }

double* SignalCDIF::TOA() { return toa; }
double* SignalCDIF::Amplitude() { return amp; }
double* SignalCDIF::Phase() { return phase; }
double* SignalCDIF::PhaseWrapped() { return phaseW; }
double* SignalCDIF::Freq() { return freq; }
double* SignalCDIF::RI() { return ri; }
double* SignalCDIF::RQ() { return rq; }
double* SignalCDIF::SI() { return si; }
double* SignalCDIF::SQ() { return sq; }
double*& SignalCDIF::Spectra() { return spectra; }
double*& SignalCDIF::SpectraTotal() { return spectraTotal; }
std::vector<double>* SignalCDIF::PeaksTOA() { return &peaksToa; }
std::vector<double>* SignalCDIF::PeaksAmp() { return &peaksAmp; }

unsigned long* SignalCDIF::SpectraCount() { return &spectraCount; }
unsigned& SignalCDIF::SpectraXRes() { return spectraXRes; }
unsigned& SignalCDIF::SpectraYRes() { return spectraYRes; }
Birch::Timespan& SignalCDIF::SpectraXOffset() { return spectraXOffset; }
Birch::Timespan& SignalCDIF::SpectraYOffset() { return spectraYOffset; }
Birch::Timespan& SignalCDIF::SpectraFreqSpan() { return spectraFreqSpan; }

template <typename T>
class Processor;

template <typename T>
struct Chunk {
public:
    enum class Stage { Free, Read, Assigned, Demodulating, Demodulated, Assembled };

    volatile Stage Status = Stage::Free;

    volatile Processor<T>* volatile proc = nullptr;

    unsigned ElementCount = 0;

    Complex<T>*      rIQ  = nullptr; // Raw IQ Data
    Complex<double>* sIQ  = nullptr; // Filtered IQ Data
    double*          pAmp = nullptr; // Partial amp calculation
    double*          pPhs = nullptr; // Unwrapped phase
    double*          wPhs = nullptr; // Wrapped phase
    double*          freq = nullptr; // FM
    double*          spec = nullptr; // Spectral data
    double           pOff = 0;

    unsigned Overlap = 0; // Number of elements which overlap the previous buffer
    unsigned Pad     = 0; // Number of elements which overlap the next buffer

    int Pos = -1;

    static unsigned DataSize() {
        return sizeof(Complex<T>) + sizeof(Complex<double>) + sizeof(double) * 5;
    }

    Chunk(unsigned size) {
        ElementCount = size;

        try {
            rIQ  = new Complex<T>[size];
            sIQ  = new Complex<double>[size];
            pAmp = new double[size];
            pPhs = new double[size];
            wPhs = new double[size];

            spec = new double[size];
            freq = new double[size];
        }
        catch (...) {
            DispError("Chunk", "Failed to allocate chunk memory!");

            delete[] rIQ;
            delete[] sIQ;
            delete[] pAmp;
            delete[] pPhs;
            delete[] wPhs;
            delete[] spec;
            delete[] freq;

            throw "Failed to allocate chunk memory!";
        }
    }
    ~Chunk() {
        delete[] rIQ;
        delete[] sIQ;
        delete[] pAmp;
        delete[] pPhs;
        delete[] wPhs;
        delete[] spec;
        delete[] freq;
    }
};

struct FIRKernel {
    Complex<double>* FreqResp = nullptr;
    unsigned int     Length   = 0;
    unsigned int     Taps     = 0;
};

template <typename T>
class Processor {
public:
    virtual volatile bool Available() = 0;
    virtual void Process(Chunk<T>* input) = 0;

    virtual ~Processor() { }
};

template <typename T>
class ProcessorCPU : public Processor<T> {
public:
    volatile bool Available() override {
        return chunk == nullptr;
    }
    void Process(Chunk<T>* input) override {
        if (!Available())
            throw "fuck";

        chunk = input;

        chunk->Status = Chunk<T>::Stage::Assigned;
    }

    ProcessorCPU(unsigned fftSize, FIRKernel* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, double sampRate, Complex<double>* tune, unsigned tuneLen, unsigned stride) {
        procThread = thread(&ProcessorCPU::process, this);

        this->size      = fftSize;
        this->filters   = filters;
        this->window    = window;
        this->dataSize  = dataSize;
        this->overlap   = overlap;
        this->sampRate  = sampRate;
        this->stride    = stride;

        try {
            if (filters != nullptr) {
                fftFiltIn  = new Complex<double>[filters->Length];
                fftFiltOut = new Complex<double>[filters->Length];
                filtPlan   = fft_make_plan(filters->Length);
            }

            //fftSpecIn  = new Complex<double>[size];
            //fftSpecOut = new Complex<double>[size];
            //specPlan   = fft_make_plan(size);
/*
            const Complex<double> zero = { 0, 0 };
            for (unsigned i = dataSize; i < fftSize; i++)
                fftSpecIn[i] = zero;
*/
            windowData = window->Build(dataSize);

            tuneBuf  = tune;
            tuneSize = tuneLen;
        } catch (...) {
            delete[] fftFiltIn;
            delete[] fftFiltOut;
            //delete[] fftSpecIn;
            //delete[] fftSpecOut;

            if (filters != nullptr)
                fft_destroy_plan(filtPlan);
            
            //fft_destroy_plan(specPlan);

            delete[] windowData;

            DispError("ProcessorCPU", "Failed to allocate fft buffers!");
            throw "Failed to allocate fft buffers!";
        }
    }
    ~ProcessorCPU() {
        halt = true;

        delete[] fftFiltIn;
        delete[] fftFiltOut;
        //delete[] fftSpecIn;
        //delete[] fftSpecOut;

        if (filters != nullptr)
            fft_destroy_plan(filtPlan);
        
        //fft_destroy_plan(specPlan);

        delete[] windowData;

        if (procThread.joinable())
            procThread.join();
    }

private:
    volatile bool halt = false;
    volatile Chunk<T>* volatile chunk = nullptr;

    Complex<double>* fftFiltIn  = nullptr;
    Complex<double>* fftFiltOut = nullptr;
    fft_plan filtPlan;

    Complex<double>* fftSpecIn  = nullptr;
    Complex<double>* fftSpecOut = nullptr;
    fft_plan specPlan;

    Complex<double>* tuneBuf = nullptr;
    unsigned tuneSize = 0;

    unsigned size;
    unsigned overlap;
    unsigned dataSize;
    WindowFunc* window = nullptr;
    FIRKernel* filters = nullptr;
    double* windowData = nullptr;
    double sampRate;
    unsigned stride;
    //unsigned pos = 0;

    void process() {
        const double frqConst = (1.0 / 360.0) * 0.000001 * sampRate;

        Stopwatch fullTime = Stopwatch();
        double actTime = 0;
        uint   chunksProcessed = 0;

        while (true) {
            if (!halt && chunk == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (halt) {
                DispInfo_("Demod Active: %.2lf% (%u chunks)", actTime / fullTime.Now() * 100, chunksProcessed);
                return;
            }

            Stopwatch timer = Stopwatch();

            //DispDebug("got: %u, had %u", chunk->Pos, pos);

            chunk->Status = Chunk<T>::Stage::Demodulating;
            chunk->pOff = 0;

            // Buffer el count is a multiple of the larger of either filter or spectra size

            // Filter data then place in sIQ
            if (filters != nullptr) {
                auto pos = chunk->rIQ; // + (chunk->Overlap - filters->Taps); // Overlap cannot be greater than taps
                auto sPos = chunk->sIQ + chunk->Overlap / 2;

                const auto inc = filters->Length - filters->Taps;
                const auto end = pos + (chunk->ElementCount);
                
                const auto inEnd   = fftFiltIn  + filters->Length;
                const auto outEnd  = fftFiltOut + filters->Length;

                for (; pos < end; pos += inc) {
                    // Copy rIQ to fftIn
                    auto cPos = pos;
                    for (auto ffi = fftFiltIn; ffi < inEnd; ffi++, cPos++) {
                        ffi->R = cPos->R; // Casting to double
                        ffi->I = cPos->I;
                    }

                    // Perform FFT
                    fft_cpx_forward(filtPlan, fftFiltIn, fftFiltOut);

                    // Convolution
                    auto fdPos = filters->FreqResp;
                    for (auto ffo = fftFiltOut; ffo < outEnd; ffo++, fdPos++)
                        *ffo = (*ffo) * (*fdPos);
                    
                    // Inverse FFT
                    fft_cpx_inverse(filtPlan, fftFiltOut, fftFiltIn);

                    // Copy the usable portion to sIQ
                    const auto sIQEnd = sPos + inc;
                    for (auto uPos = fftFiltIn + filters->Taps; sPos < sIQEnd; sPos++, uPos++) {
                        *sPos = *uPos;// * ampMod;
                    }
                }
            } else { // No filters, simply copy data over
                const Complex<T>* const end = chunk->rIQ + chunk->ElementCount;
                auto sb = chunk->sIQ;

                for (const Complex<T>* it = chunk->rIQ; it < end; ++it, ++sb) {
                    sb->R = it->R; // Casting to double
                    sb->I = it->I;
                }
            }

            if (tuneBuf != nullptr) {
                unsigned tIdx = (chunk->Pos * (chunk->ElementCount - chunk->Overlap)) % tuneSize;
                const Complex<double>* const end = chunk->sIQ + chunk->ElementCount;
                for (Complex<double>* it = chunk->sIQ; it < end; ++it) {
                    const auto v = tuneBuf[tIdx++ % tuneSize].R;
                    *it = *it * Complex<double>(cos(v), sin(v));
                }
            }

            // Spectra here
            //for (unsigned i = 0; i < chunk->ElementCount; i++)
            //    chunk->spec = 0;

            // Amplitude, phase, freq
            const auto end = chunk->sIQ + chunk->ElementCount - chunk->Pad;
            auto sIQ = chunk->sIQ + chunk->Overlap;
            auto rIQ = chunk->rIQ + chunk->Overlap;
            auto amp = chunk->pAmp + chunk->Overlap;
            auto phs = chunk->pPhs + chunk->Overlap;
            auto wphs = chunk->wPhs + chunk->Overlap;
            auto frq = chunk->freq + chunk->Overlap;

            double lastPhs = 0;

            for (; sIQ < end; sIQ++, amp++, phs++, frq++, wphs++, rIQ++) {
                const auto iq = *sIQ;

                *amp = iq.R * iq.R + iq.I * iq.I;
                
                // TODO: Radians as option
                auto phase = (atan2(iq.I, iq.R) * (180.0 / M_PI));

                //auto phase = acos(iq.R / sqrt(*amp)) * (180.0 / M_PI);

                *wphs = phase;
                *phs = phase + chunk->pOff;


                // Unwrap phase
                auto pDif = *phs - lastPhs;
                if (pDif > 180) {
                    *phs -= 360;
                    chunk->pOff -= 360;

                    pDif = *phs - lastPhs;
                } else if (pDif < -180) {
                    *phs += 360;
                    chunk->pOff += 360;

                    pDif = *phs - lastPhs;
                }
                
                *frq = pDif * frqConst;

                lastPhs = *phs;
            }

            //DispDebug("done: %u", chunk->Pos);

            chunk->Status = Chunk<T>::Stage::Demodulated;

            chunk = nullptr;

            actTime += timer.Now();
            chunksProcessed++;
        }

    }

    thread procThread;
};

#ifdef FFT_IMPL_CUDA
#include "cudaCDIF.cuh"
#endif

CDIFReader::CDIFReader(unsigned cacheSize) {
    cacheTotalSize  = cacheSize;
    threadCount     = thread::hardware_concurrency();

    if (threadCount < 4)
    {
        DispWarning("CDIFReader", "You do not have enough processor cores to meet the minimum requirements.");
    }
    else
    {
        threadCount -= 2;
    }

#ifdef FFT_IMPL_CUDA
    threadCount += cuda_count_devices();
#endif

    cacheCount      = threadCount * 2;
    threadCacheSize = cacheTotalSize / cacheCount;

    //threadCacheSize = 256 * 1024 * 1024; // 256MB
    //cacheCount      = cacheSize / threadCacheSize;
}
CDIFReader::~CDIFReader() {
    delete[] memory;
}

template<typename T>
void CDIFReader::process(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, unsigned stride, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned specRes, volatile float* progress, volatile bool* stop) {
    Stopwatch setupTimer = Stopwatch();
    
    if (stride != 1)
        stride *= 2;

    // Find max fft and fir tap count
    auto maxFFT   = fftSize;
    auto filtFFT  = 0u;
    auto filtTaps = 0u;

    for (auto f : *filters) {
        if (f->FFTSize() > filtFFT)
            filtFFT = f->FFTSize();

        if (f->TapCount() > filtTaps)
            filtTaps = f->TapCount();
    }
    
    if (filtFFT > fftSize)
        maxFFT = fftSize;


    bool calcPeaks = false;

    // Build filter kernel if filters are used
    Complex<double>* filterKernel = nullptr;
    FIRKernel* fir = nullptr;

    if (filters->size() > 0) {
        fir = new FIRKernel();
        fir->Length   = filtFFT;
        fir->Taps     = filtTaps;

        filterKernel = new Complex<double>[filtFFT];
        Complex<double>* curKernelIn  = new Complex<double>[filtFFT];
        Complex<double>* curKernelOut = new Complex<double>[filtFFT];

        fir->FreqResp = filterKernel;

        fft_plan fkPlan = fft_make_plan(filtFFT);

        const Complex<double> zero = { 0, 0 };
        unsigned it = 0;
        for (auto f : *filters) {
            // Setup input buffer
            const auto ir = f->ImpulseResponse(); // Window will have already been applied
            for (unsigned i = 0; i < f->TapCount(); i++)
                curKernelIn[i] = ir[i];

            // Zero out remaining buffer length
            for (unsigned i = f->TapCount(); i < filtFFT; i++)
                curKernelIn[i] = zero;

            // Send the first fft to the main buffer
            if (it == 0) {
                fft_cpx_forward(fkPlan, curKernelIn, filterKernel);

                if (f->Shift())
                    fftshift(filterKernel, filtFFT);
            } else {
                fft_cpx_forward(fkPlan, curKernelIn, curKernelOut);

                if (f->Shift())
                    fftshift(curKernelOut, filtFFT);

                // Add this filter kernel to output kernel
                for (unsigned i = 0; i < filtFFT; i++)
                    filterKernel[i] = filterKernel[i] * curKernelOut[i];
            }

            if (f->Peaks())
                calcPeaks = true;

            it++;
        }

        delete[] curKernelIn;
        delete[] curKernelOut;

        fft_destroy_plan(fkPlan);
    }

    auto* const peaksToa = signal->PeaksTOA();
    auto* const peaksAmp = signal->PeaksAmp();

    peaksToa->clear();
    peaksAmp->clear();

    // Sanitize timespan
    const auto fileSpan = getter->Time();
    if (span.Start < fileSpan.Start)
        span.Start = fileSpan.Start;
    if (span.End > fileSpan.End)
        span.End = fileSpan.End;

    signal->Time()   = span;
    signal->Stride() = stride;

    // Setup constants
    const auto elCount   = getter->SpanElementCount(span);
    const auto maxPerBuf = threadCacheSize / (sizeof(Chunk<T>) + Chunk<T>::DataSize());
    const auto elsPerBuf = (maxPerBuf / maxFFT) * maxFFT; // Integer multiplication here performs a floor()
    
    auto padPerBuf = (filtFFT - filtTaps) - (elsPerBuf - ((filtFFT - filtTaps) * (elsPerBuf / ((filtFFT - filtTaps) == 0 ? 1 : (filtFFT - filtTaps))))); // Number of extra elements to add to the end of the buffer in order to allow filtering
    if (filters->size() == 0)
        padPerBuf = 0;

    const auto reqBuffs  = static_cast<unsigned>(ceil(elCount / (double)(elsPerBuf - overlap)));
    const auto startEl   = getter->TimeIdx(span.Start);
    //const auto endEl     = startEl + getter->SpanElementCount(span);
    const auto itMax     = UINT32_MAX;
    const auto itBufs    = static_cast<unsigned>(ceil(elCount / static_cast<double>(itMax)));
    const auto sampRate  = getter->SampleRate();
    const auto bufSize   = (elsPerBuf + padPerBuf) * (sizeof(T) * 2);
    const auto epbPadded = elsPerBuf + padPerBuf;

    DispInfo_("Total Chunks: %u", reqBuffs);

    //printf("%lf\n", span.Start);

    //DispDebug("reqBufs: %u", reqBuffs);
    //DispDebug("epb: %lu", epbPadded);
    //DispDebug("itBufs: %u", itBufs);
    //DispDebug("pad: %lu", padPerBuf);
    //DispDebug("buffer bytes: %lu", bufSize);
    //DispDebug("max bytes: %u", threadCacheSize);
    //DispDebug("stride: %u", stride);
    //DispDebug("el count: %llu", elCount);
    //DispDebug("tune: %lf", tune);

    // This should be resolved prior to this function being called
    if (elsPerBuf == 0) {
        DispError("CDIFReader", "FFT Size was too high given the amount of memory available!");
        return;
    }

    unsigned tuneSize = 0;
    Complex<double>* tuneBuf = nullptr;

    if (tune != 0) {
        tuneSize = sampRate;
        tuneBuf = new Complex<double>[tuneSize];

        const double tuneStep = 1.0 / sampRate;
        const double tc = 2 * M_PI * tune;
        double tv = 0;
        for (unsigned i = 0; i < tuneSize; i++) {
            const double tt = tv * tc;
            tuneBuf[i] = { tt, tt };
            tv += tuneStep;
        }
    }

    // Verify progress and stop pointers
    float progDummy = 0;
    bool  stopDummy = false;

    if (progress == nullptr) {
        DispWarning("CDIFReader", "Progress output is null!");
        progress = &progDummy;
    }
    if (stop == nullptr) {
        DispWarning("CDIFReader", "Stop flag is null! (Nothing can stop me now)");
        stop = &stopDummy;
    }

    *progress = 0;

    // Buffer to hold processors
    vector<Processor<T>*> procs = vector<Processor<T>*>();

    // Init processors
    const unsigned cudaCount = 0; //cuda_count_devices();
    for (unsigned i = 0; i < threadCount - cudaCount; i++)
        procs.push_back(new ProcessorCPU<T>(fftSize, fir, window, dataSize, overlap, sampRate, tuneBuf, tuneSize, stride));

    //for (unsigned i = 0; i < cudaCount; i++)
    //    procs.push_back(new ProcessorCUDA<T>());

    // Buffer to hold all chunks in the order they appear in the file
    vector<volatile Chunk<T>*> chunkStream = vector<volatile Chunk<T>*>(reqBuffs, nullptr);


    // Initialize memory
    vector<Chunk<T>*> chunks = vector<Chunk<T>*>();
    for (unsigned i = 0; i < cacheCount; i++)
        chunks.push_back(new Chunk<T>(bufSize));
    
    // Disk IO / Chunk assignment thread
    // Will read until either the end of the required portion or stop is set to true
    // Will assign read chunks to the next free processor
    thread readThread = thread([&]() {
        const auto beginOvr  = overlap + filtTaps; // Still have to filter the overlap portion
        const auto endOvr    = padPerBuf;
        const auto bufferInc = elsPerBuf - beginOvr;

        unsigned itCur = 0;

        // chunkStream index
        unsigned pos = 0;
        // last assigned chunk index
        unsigned aPos = 0;
        // file element index
        unsigned elPos = (double)startEl - beginOvr < 0 ? 0 : (startEl - beginOvr);
        unsigned lastElPos = elPos;

        uint noChunks = 0;
        uint noProcs  = 0;

        while (aPos < reqBuffs && !*stop) {
            // Get next available chunk
            Chunk<T>* readBuf = nullptr;
            bool skipProc = false;

            bool allProcs = true;

            // Set chunk to be read or assign to next available processor
            for (auto chunk : chunks) {
                if (readBuf == nullptr && chunk->Status == Chunk<T>::Stage::Free) {
                    readBuf = chunk;
                }
                
                if (chunk->Status == Chunk<T>::Stage::Read) {
                    skipProc = true;
                    for (Processor<T>* p : procs) {
                        if (p->Available()) {
                            chunk->proc = p;

                            p->Process(chunk);

                            //DispDebug("sent: %u", chunk->Pos);

                            aPos++;

                            skipProc = false;

                            allProcs = false;

                            break;
                        }
                    }
                }
            }

            if (allProcs) {
                noProcs++;

                if (noProcs == 99999) {
                    DispDebug("proc: %u", chunk->Pos);
                }
            }

            // All chunks are currently in use
            if (readBuf == nullptr) {
                //DispInfo_("All chunks in use");
                noChunks++;
                continue;
            }
            
            if (pos >= reqBuffs)
                continue;

            if (!getter->Seek(elPos + itCur * itMax)) {
                DispError("CDIFReader::IO", "Seek failed!");
                *stop = true;
                break;
            }

            // The overlap period should already be cached since we just read it
            // It will probably be slower to add more caching here but I haven't tested

            // Need to fill the overlap portion with zeros if reading from the beginning of the file
            if (elPos == 0) {
                const unsigned padLen = fabs((double)startEl - (double)beginOvr);
                const Complex<T> zero = { 0, 0 };

                for (unsigned i = 0; i < padLen; i++) {
                    readBuf->rIQ[i] = zero;
                }

                readBuf->ElementCount = getter->Read(reinterpret_cast<char*>(readBuf->rIQ + padLen), epbPadded - padLen) + padLen;
            } else {
                readBuf->ElementCount = getter->Read(reinterpret_cast<char*>(readBuf->rIQ), epbPadded);
            }

/*
            // Pad the last buffer with zeros
            if (readBuf->ElementCount < epbPadded) {
                const unsigned padLen = epbPadded - readBuf->ElementCount;
                const Complex<T> zero = { 0, 0 };

                for (unsigned i = 0; i < padLen; i++)
                    readBuf->rIQ[readBuf->ElementCount + i] = zero;
                
                readBuf->ElementCount += padLen;
            }
*/
            readBuf->Overlap = beginOvr;
            readBuf->Pad = padPerBuf;
            elPos += bufferInc;

            if (elPos < lastElPos) {
                //DispDebug("overflow %u", itCur);
                itCur++;
            }

            lastElPos = elPos;

            //DispDebug("read: %u", pos);

            // Prepare chunk for processing
            readBuf->Status = Chunk<T>::Stage::Read;
            readBuf->Pos = pos;

            chunkStream[pos] = readBuf;

            pos++;

            // Sleep
            //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        DispInfo_("No Chunks: %u", noChunks);
        DispInfo_("No Procs:  %u", noProcs);
    });

    // May the bridges I've burnt light my way


    // Assembly
    unsigned currentChunk  = 0;
    unsigned currentStride = 0;
    unsigned currentPos    = 0;

    double* const si = signal->SI();
    double* const sq = signal->SQ();
    double* const ri = signal->RI();
    double* const rq = signal->RQ();
    double* const am = signal->Amplitude();
    double* const pm = signal->Phase();
    double* const wp = signal->PhaseWrapped();
    double* const fm = signal->Freq();
    double* const ta = signal->TOA();
    double* const sp = signal->Spectra();

    unsigned long* const ec = signal->ElementCount();

    const double timingMult = (1.0 / sampRate) * (stride > 1 ? (stride / 2) : 1); // todo: Allow changing TOA from usec
    const double toaOffset  = span.Start;

    Span<double> canSI = 0;
    Span<double> canSQ = 0;
    Span<double> canRI = 0;
    Span<double> canRQ = 0;
    Span<double> canAM = 0;
    Span<double> canPM = 0;
    Span<double> canOF = 0; // pOff
    Span<double> canWP = 0;
    Span<double> canFM = 0;

    double pOff = 0;
    bool reset = false;
    double lastPhs = 0;

    //DispDebug("Setup Time: %lf", setupTimer.Now());

    const double peakThres   = 70000;
    const double peakThresSq = pow(peakThres, 2);
    bool recPeak = false;
    unsigned long long peakStart = 0;
    unsigned long long peakEl    = 0;
    double peakMax = 0;

    Stopwatch totalTime = Stopwatch();
    double waitTime = 0;

    while (!*stop && currentChunk < reqBuffs) {
        Stopwatch waitTimer = Stopwatch();

        const auto chunk = chunkStream[currentChunk];

        if (chunk == nullptr || chunk->Status != Chunk<T>::Stage::Demodulated) {
            waitTime += waitTimer.Now();
            continue;
        }
        
        //DispDebug("ass: %u", currentChunk);

        auto cOvr = chunk->Overlap;

        // Unwrap phase
        double pDif = (chunk->wPhs[cOvr] - lastPhs);
        if (pDif > 180) {
            //chunk->pPhs[0] -= 360;
            //chunk->pOff -= 360;

            pDif -= 360;
        } else if (pDif < -180) {
            //chunk->pPhs[0] += 360;
            //chunk->pOff += 360;

            pDif += 360;
        }

        //DispDebug("%lf", pDif);

        //chunk->freq[0] = (sampRate * pDif) / 360 * 0.000001;

        if (chunk->Overlap > 0) {
            //chunk->freq[0] = (sampRate * pDif) / 360 * 0.000001;
            chunk->freq[cOvr] = chunk->freq[cOvr + 1];
        }

        for (unsigned i = cOvr; i < chunk->ElementCount - padPerBuf; i++) {
            auto& curSIQ  = chunk->sIQ[i];
            auto& curRIQ  = chunk->rIQ[i];
            auto& curpPhs = chunk->pPhs[i];
            auto& curwPhs = chunk->wPhs[i];
            auto& curpAmp = chunk->pAmp[i];
            auto& curFreq = chunk->freq[i];

            if (reset) {
                canSI = curSIQ.R;
                canSQ = curSIQ.I;
                canRI = curRIQ.R;
                canRQ = curRIQ.I;
                canPM = curpPhs;
                canOF = { pOff, pOff };
                canWP = curwPhs;
                canAM = curpAmp;
                canFM = curFreq;

                reset = false;
            } else {
                
                if (curpAmp < canAM.Start)
                    canAM.Start = curpAmp;
                if (curSIQ.R < canSI.Start)
                    canSI.Start = curSIQ.R;
                if (curSIQ.I < canSQ.Start)
                    canSQ.Start = curSIQ.I;
                if (curRIQ.R < canRI.Start)
                    canRI.Start = curRIQ.R;
                if (curRIQ.I < canRQ.Start)
                    canRQ.Start = curRIQ.I;
                if (curpPhs < canPM.Start) {
                    canPM.Start = curpPhs;
                    canOF.Start = pOff;
                }
                if (curwPhs < canWP.Start)
                    canWP.Start = curwPhs;
                if (curFreq < canFM.Start)
                    canFM.Start = curFreq;


                if (curpAmp > canAM.End)
                    canAM.End = curpAmp;
                if (curSIQ.R > canSI.End)
                    canSI.End = curSIQ.R;
                if (curSIQ.I > canSQ.End)
                    canSQ.End = curSIQ.I;
                if (curRIQ.R > canRI.End)
                    canRI.End = curRIQ.R;
                if (curRIQ.I > canRQ.End)
                    canRQ.End = curRIQ.I;
                if (curpPhs > canPM.End) {
                    canPM.End = curpPhs;
                    canOF.End = pOff;
                }
                if (curwPhs > canWP.End)
                    canWP.End = curwPhs;
                if (curFreq > canFM.End)
                    canFM.End = curFreq;
                
            }

            if (++currentStride == stride) {
                const double toa = currentPos * timingMult + toaOffset;

                si[currentPos] = canSI.Start;
                sq[currentPos] = canSQ.Start;
                ri[currentPos] = canRI.Start;
                rq[currentPos] = canRQ.Start;
                am[currentPos] = sqrt(canAM.Start);
                pm[currentPos] = canPM.Start + canOF.Start;
                wp[currentPos] = canWP.Start;
                fm[currentPos] = canFM.Start;
                ta[currentPos] = toa;

                currentPos++;

                if (stride != 1) {
                    si[currentPos] = canSI.End;
                    sq[currentPos] = canSQ.End;
                    ri[currentPos] = canRI.End;
                    rq[currentPos] = canRQ.End;
                    am[currentPos] = sqrt(canAM.End);
                    pm[currentPos] = canPM.End + canOF.End;
                    wp[currentPos] = canWP.End;
                    fm[currentPos] = canFM.End;
                    ta[currentPos] = toa;

                    currentPos++;
                }

                *ec = currentPos;
                currentStride = 0;

                reset = true;
            }
            
/*
            if (calcPeaks) {
                if (recPeak && chunk->pAmp[i] > peakMax)
                    peakMax = chunk->pAmp[i];

                if (!recPeak && chunk->pAmp[i] > peakThresSq) {
                    recPeak = true;
                    peakStart = peakEl;
                    peakMax = chunk->pAmp[i];
                } else if (recPeak && chunk->pAmp[i] < peakThresSq) {
                    recPeak = false;
                    
                    const double cEl = peakEl + ((peakEl - peakStart) / 2);
                    peaksToa->push_back(cEl * (1.0 /sampRate) + toaOffset);
                    peaksAmp->push_back(sqrt(peakMax));
                }

                peakEl++;
            }
*/
            
            if (currentPos >= signal->Size() || currentPos >= elCount)
                goto end;
        }

        lastPhs = chunk->wPhs[chunk->ElementCount - 1];

        pOff += chunk->pOff;

        chunkStream[currentChunk] = nullptr;
        chunk->Status = Chunk<T>::Stage::Free;
        currentChunk++;

        *progress = (float)currentChunk / reqBuffs;;
    }

    end:
    //*(signal->ElementCount()) = currentPos - 1;
    
    //*stop = true;

    //DispDebug("ec: %lu", *ec);

    DispInfo_("Assembly Inactive: %.2lf%", waitTime / totalTime.Now() * 100);

    for (int i = 0; i < procs.size(); i++) {
        delete procs[i];
        procs[i] = nullptr;
    }

    if (readThread.joinable())
        readThread.join();

    for (auto& c : chunks)
        delete c;

    delete[] filterKernel;
    delete   fir;

    delete[] tuneBuf;
}

void CDIFReader::Process(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, unsigned stride, double tune, vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned specRes, volatile float* progress, volatile bool* stop) {
    if (working) {
        DispError("CDIFReader", "Cannot start read. Read is already in progress.");
        return;
    }
    
    //printf("%lf->%lf\n", span.Start, span.End);

    auto timer = Stopwatch();

    working = true;

    switch (getter->Format()) {
        case Birch::DataFormat::Int8:
            process<int8_t>(getter,   signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
        case Birch::DataFormat::Int16:
            process<int16_t>(getter,  signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
        case Birch::DataFormat::Int32:
            process<int32_t>(getter,  signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
        case Birch::DataFormat::Int64:
            process<int64_t>(getter,  signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
        case Birch::DataFormat::Float:
            process<float_t>(getter,  signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
        case Birch::DataFormat::Double:
            process<double_t>(getter, signal, span, stride, tune, filters, window, dataSize, overlap, fftSize, specRes, progress, stop);
            break;
    }

    working = false;

    DispInfo("CDIFReader", "Read Time: %lfs", timer.Now());
}

template<typename T>
struct SpectraBuffer {
    Complex<T>* IQ        = nullptr;
    unsigned    ElCount   = 0;
    unsigned    Offset    = 0;
    unsigned    OffsetIts = 0;

    const unsigned ItSize = UINT32_MAX;

    enum class Stage { Free, Read, Assigned };
    volatile Stage Status = Stage::Free;

    explicit SpectraBuffer(unsigned elCount) {
        ElCount = elCount;
        IQ      = new Complex<T>[elCount];
    }
    ~SpectraBuffer() {
        delete[] IQ;
    }
    
    SpectraBuffer (const SpectraBuffer&) = delete;
    SpectraBuffer& operator= (const SpectraBuffer&) = delete;
};

template <typename T>
class SpectraProcessor {
public:
    virtual bool Available() = 0;
    virtual void Process(SpectraBuffer<T>* input) = 0;

    virtual ~SpectraProcessor() { }
};

template<typename T>
class SpectraProcessorCPU : public SpectraProcessor<T> {
public:
    bool Available() override {
        return input == nullptr;
    }

    SpectraProcessorCPU(volatile double* output, unsigned fftSize, unsigned dataSize, unsigned overlap, unsigned bins, unsigned frqs, unsigned ffts, const double* winBuf, const int* ffMap, float ffScale, fft_plan plan, unsigned long long totalElCount)
                       : output(output), fftSize(fftSize), dataSize(dataSize), overlap(overlap), bins(bins), frqs(frqs), ffts(ffts), winBuf(winBuf), ffMap(ffMap), ffScale(ffScale), plan(plan), totalElCount(totalElCount)
    {
        fftIn  = new Complex<double>[fftSize];
        fftOut = new Complex<double>[fftSize];

        const auto zero = Complex<double>(0, 0);
        for (unsigned i = dataSize; i < fftSize; i++)
            fftIn[i] = zero;

        procThread = thread(&SpectraProcessorCPU::process, this);
    }
    ~SpectraProcessorCPU() {
        stop = true;

        if (procThread.joinable())
            procThread.join();

        delete[] fftIn;

        delete[] fftOut;
    }

    void Process(SpectraBuffer<T>* input) override {
        if (!Available())
            throw "fuck";
        
        this->input = input;
        input->Status = SpectraBuffer<T>::Stage::Assigned;
    }

    SpectraProcessorCPU (const SpectraProcessorCPU&) = delete;
    SpectraProcessorCPU& operator= (const SpectraProcessorCPU&) = delete;
private:
    fft_plan plan    = nullptr;

    const double* winBuf;
    const int*    ffMap;

    Complex<double>* fftIn  = nullptr;
    Complex<double>* fftOut = nullptr;

    volatile double* const output;

    const float    ffScale;
    const unsigned fftSize;
    const unsigned dataSize;
    const unsigned overlap;
    const unsigned bins;
    const unsigned frqs;
    const unsigned ffts;

    const unsigned long long totalElCount;

    SpectraBuffer<T>* input = nullptr;

    volatile bool stop = false;

    thread procThread;

    void process() {
        double scaleFactor = (double)bins / totalElCount;

        while (true) {
            if (!stop && input == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (stop)
                return;
            
            unsigned long long currentEl = input->Offset + input->OffsetIts * (unsigned long long)input->ItSize;
            const Complex<T>* const iq = input->IQ;

            unsigned cBin = 0;

            //printf("cBin: %u\n", cBin);

            for (unsigned i = 0; i < ffts; i++) {
                if (stop)
                    return;
                
                cBin = currentEl * scaleFactor;

                //printf("cBin: %u\n", cBin);

                // Buffer can be longer than what we're processing
                // Quit if at end of proc interval
                if (cBin >= bins)
                    break;
                
                const unsigned sbOffset = cBin * frqs;

                // Copy iq to fft buffer and apply window
                // Required as fft only takes float as input, so conversion is made here
                const int iqOffset = i * (dataSize - overlap);
                for (int j = 0; j < dataSize; j++)
                {
                    fftIn[j].I = iq[iqOffset + j].I * winBuf[j];
                    fftIn[j].R = iq[iqOffset + j].R * winBuf[j];
                }

                fft_cpx_forward(plan, fftIn, fftOut);

                // Copy to output buffer
                volatile double *const oBuf = output + sbOffset;
                for (int j = 0; j < fftSize; j++)
                {
                    const float samp = fftOut[j].I * fftOut[j].I + fftOut[j].R * fftOut[j].R;
                    volatile double *const cur = oBuf + ffMap[j];

                    if (*cur < samp)
                        *cur = samp;
                }

                currentEl += dataSize - overlap;
            }

            input->Status = SpectraBuffer<T>::Stage::Free;
            input = nullptr;
        }
    }
};

/*
#ifdef FFT_IMPL_CUDA
class SpectraProcessorGPU : public SpectraProcessor<T> {
public:
    bool Available() override {
        return input == nullptr;
    }

    SpectraProcessorGPU(volatile double* output, unsigned fftSize, unsigned dataSize, unsigned overlap, unsigned bins, unsigned frqs, unsigned ffts, const double* winBuf, const int* ffMap, float ffScale, fft_plan plan, unsigned long long totalElCount)
                       : output(output), fftSize(fftSize), dataSize(dataSize), overlap(overlap), bins(bins), frqs(frqs), ffts(ffts), winBuf(winBuf), ffMap(ffMap), ffScale(ffScale), plan(plan), totalElCount(totalElCount)
    {
        fftIn  = new Complex<double>[fftSize];
        fftOut = new Complex<double>[fftSize];

        const auto zero = Complex<double>(0, 0);
        for (unsigned i = dataSize; i < fftSize; i++)
            fftIn[i] = zero;

        procThread = thread(&SpectraProcessorGPU::process, this);
    }
    ~SpectraProcessorGPU() {
        stop = true;

        if (procThread.joinable())
            procThread.join();

        delete[] fftIn;

        delete[] fftOut;
    }

    void Process(SpectraBuffer<T>* input) override {
        if (!Available())
            throw "fuck";
        
        this->input = input;
        input->Status = SpectraBuffer<T>::Stage::Assigned;
    }

    SpectraProcessorGPU (const SpectraProcessorGPU&) = delete;
    SpectraProcessorGPU& operator= (const SpectraProcessorGPU&) = delete;
private:
    fft_plan plan    = nullptr;

    const double* winBuf;
    const int*    ffMap;

    Complex<double>* fftIn  = nullptr;
    Complex<double>* fftOut = nullptr;

    volatile double* const output;

    const float    ffScale;
    const unsigned fftSize;
    const unsigned dataSize;
    const unsigned overlap;
    const unsigned bins;
    const unsigned frqs;
    const unsigned ffts;

    const unsigned long long totalElCount;

    SpectraBuffer<T>* input = nullptr;

    volatile bool stop = false;

    thread procThread;

    void process() {
        double scaleFactor = (double)bins / totalElCount;

        while (true) {
            if (!stop && input == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (stop)
                return;
            
            unsigned long long currentEl = input->Offset + input->OffsetIts * (unsigned long long)input->ItSize;
            const Complex<T>* const iq = input->IQ;

            unsigned cBin = 0;

            //printf("cBin: %u\n", cBin);

            for (unsigned i = 0; i < ffts; i++) {
                if (stop)
                    return;
                
                cBin = currentEl * scaleFactor;

                //printf("cBin: %u\n", cBin);

                // Buffer can be longer than what we're processing
                // Quit if at end of proc interval
                if (cBin >= bins)
                    break;
                
                const unsigned sbOffset = cBin * frqs;

                // Copy iq to fft buffer and apply window
                // Required as fft only takes float as input, so conversion is made here
                const int iqOffset = i * (dataSize - overlap);
                for (int j = 0; j < dataSize; j++)
                {
                    fftIn[j].I = iq[iqOffset + j].I * winBuf[j];
                    fftIn[j].R = iq[iqOffset + j].R * winBuf[j];
                }

                fft_cpx_forward(plan, fftIn, fftOut);

                // Copy to output buffer
                volatile double *const oBuf = output + sbOffset;
                for (int j = 0; j < fftSize; j++)
                {
                    const float samp = fftOut[j].I * fftOut[j].I + fftOut[j].R * fftOut[j].R;
                    volatile double *const cur = oBuf + ffMap[j];

                    if (*cur < samp)
                        *cur = samp;
                }

                currentEl += dataSize - overlap;
            }

            input->Status = SpectraBuffer<T>::Stage::Free;
            input = nullptr;
        }
    }
};
#endif
*/
 
template<typename T>
void CDIFReader::spectra(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, Birch::Span<double> freqSpan, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned res, volatile float *progress, volatile bool *stop) {
    auto setupTimer = Stopwatch();
    
    const double bandwidth = signal->SampleRate() / 1000000;
    const double freqRes   = bandwidth / fftSize;

    freqSpan.Clamp(Span<double>(-bandwidth / 2, bandwidth / 2));
    span.Clamp(signal->Time());

    const unsigned numFreqs = [&]() {
        unsigned c = fabs(freqSpan.Length()) / freqRes + 1;
        if (c > fftSize) return fftSize;
        return c;
    }();


    // Frequency offsets for image
    const double flOff = (fabs(freqSpan.Start) / freqRes - floor(fabs(freqSpan.Start) / freqRes)) * freqRes;
    const double fhOff = (fabs(freqSpan.End)   / freqRes - floor(fabs(freqSpan.End)   / freqRes)) * freqRes;

    const double fbrScale = fftSize / bandwidth;

    // FFT bin offsets for fft to bin scaling
    const int flOffB = floor(freqSpan.Start * fbrScale + fftSize / 2.);
    const int fhOffB = ceil(freqSpan.End    * fbrScale + fftSize / 2.);
    
    //const double ovrCor = overlap * signal->SampleInterval();
    //span.End += ovrCor;

    // Adjust time to be a multiple of the fft size
    const double overtime = (dataSize - getter->SpanElementCount(span) % dataSize) * signal->SampleInterval();
    span.End += overtime;

    signal->SpectraXOffset() = Timespan(0, overtime);
    signal->SpectraYOffset() = Timespan(flOff, fhOff);


    const auto     elCount   = getter->SpanElementCount(span);
    const unsigned binCount  = elCount / (dataSize - overlap);
    const unsigned elsPerBin = elCount / binCount;

    if (binCount == 0) {
        DispError("CDIFReader (Spectra)", "No data to process!");
        return;
    }

    const unsigned dataBytes = dataSize * sizeof(Complex<T>);
    const unsigned readBufferSize = floor(threadCacheSize / dataBytes) * dataBytes;

    if (readBufferSize < dataBytes) {
        DispError("CDIFReader (Spectra)", "Data size > cache size!");
        return;
    }

    // Number of ffts to be performed per bin
    const unsigned elsPerBuffer = readBufferSize / sizeof(Complex<T>);
    const unsigned fftCnt = (elsPerBuffer - dataSize) / (dataSize - overlap);

    const unsigned viewFreqBins = fhOffB - flOffB;

    // Will generate a rex x res image
    const unsigned maxBins = binCount     > res ? res : binCount;
    const unsigned maxFrqs = viewFreqBins > res ? res : viewFreqBins;


    signal->SpectraXRes() = maxFrqs;
    signal->SpectraYRes() = maxBins;

    if (*signal->SpectraCount() != maxBins * maxFrqs || signal->Spectra() == nullptr) {
        delete[] signal->Spectra();
        signal->Spectra() = new double[maxBins * maxFrqs];
    }

    *signal->SpectraCount() = 0;

    memset(signal->Spectra(), 0, maxBins * maxFrqs * sizeof(double));


    const auto startEl      = getter->TimeIdx(span.Start);
    const auto elsOutPerBuf = elCount / (maxBins * maxFrqs);

    *progress = 0;

    signal->Time() = span;
    signal->SpectraFreqSpan() = freqSpan;


    // Create buffers
    vector<SpectraBuffer<T>*> buffers;
    for (unsigned i = 0; i < cacheCount; i++)
        buffers.push_back(new SpectraBuffer<T>(elsPerBuffer));

    fft_plan cpuPlan = fft_make_plan(fftSize);
    auto winBuf = window->Build(dataSize);

    // Maps fft freq bin (col) to image bin
    auto ffMap = new int[fftSize];
    auto ffScale = (float)maxFrqs / fftSize;

    for (int i = 0; i < fftSize / 2; i++)
        ffMap[i] = ffScale * (i + (fftSize -1) / 2);
    for (int i = fftSize / 2; i < fftSize; i++)
        ffMap[i] = ffScale * (i - (fftSize -1) / 2);


    // Processing threads
    vector<SpectraProcessor<T>*> processors;
    for (unsigned i = 0; i < threadCount; i++)
        processors.push_back(new SpectraProcessorCPU<T>(signal->Spectra(), fftSize, dataSize, overlap, maxBins, maxFrqs, fftCnt, winBuf, ffMap, ffScale, cpuPlan, elCount));

    const unsigned reqBufs = ceil((double)elCount / (double)elsPerBuffer);
    unsigned bufsRead = 0;
    unsigned bufsProc = 0;

    const unsigned itMax = UINT32_MAX;

    unsigned elPos     = startEl;
    unsigned lastElPos = elPos;
    unsigned itCur     = floor(startEl / itMax);
    unsigned startElU  = elPos;

    //DispDebug("Setup time: %lfs", setupTimer.Now());

    while (!*stop && bufsProc < reqBufs) {
        // Get next available buffer
        SpectraBuffer<T>* buf = nullptr;
        bool skipProc = false;

        // Set buffer to be read or assign to next available processor
        for (auto b : buffers) {
            if (b->Status == SpectraBuffer<T>::Stage::Free) {
                buf = b;
            } else if (b->Status == SpectraBuffer<T>::Stage::Read && !skipProc) {
                skipProc = true;

                for (auto p : processors) {
                    if (p->Available()) {
                        p->Process(b);

                        *signal->SpectraCount() = elsOutPerBuf * bufsProc;
                        *progress = (float)bufsProc / reqBufs;

                        bufsProc++;

                        skipProc = false;
                        break;
                    }
                }
            }
        }

        // All buffers are in use
        if (buf == nullptr)
            continue;
        
        // Finished reading but not processing
        if (bufsRead >= reqBufs)
            continue;
        
        if (!getter->Seek(elPos + itCur * itMax)) {
            DispError("CDIFReader::IO (Spectra)", "Seek failed!");
            *stop = true;
            break;
        }

        auto read = getter->Read(reinterpret_cast<char*>(buf->IQ), elsPerBuffer);
        if (read < elsPerBuffer) {
            const Complex<T> zero(0, 0);
            for (unsigned i = read; i < elsPerBuffer; i++)
                buf->IQ[i] = zero;
        }

        buf->Offset = elPos - startElU;
        buf->OffsetIts = itCur;

        lastElPos = elPos;

        elPos += (dataSize - overlap) * fftCnt;

        if (elPos < lastElPos)
            itCur++;

        buf->Status = SpectraBuffer<T>::Stage::Read;
        
        bufsRead++;
    }

    if (!*stop) {
        // Wait for processors to finish
        while (true) {
            bool allDone = true;

            for (auto p : processors) {
                if (!p->Available()) {
                    allDone = false;
                    break;
                }
            }

            if (allDone)
                break;
        }
    }

    auto killTimer = Stopwatch();

    *signal->SpectraCount() = maxBins * maxFrqs;
/*
    {
        unsigned colors[256];
        for (int i = 0; i < 256; i++) {
            auto c = i + 24;
            if (c > 255)
                c = 255;

            colors[i] =  (uint)c / 4;
            colors[i] |= (uint)c     << 8;
            colors[i] |= (uint)c / 2 << 16;
            colors[i] |= (uint)(255) << 24;
        }

        float maxMagI = 0;
        const auto sb = signal->Spectra();
        const auto sc = *signal->SpectraCount();

        for (int i = 0; i < sc; i++)
            if (sb[i] > maxMagI)
                maxMagI = sb[i];

        const float scale = 255 / (maxMagI / 10.0);

        unsigned* img = new unsigned[(unsigned)sc];

        for (int i = 0; i < sc; i++) {
            img[i] = colors[(unsigned char)(sb[i] * scale)];
        }
        
        stbi_write_png("out.png", signal->SpectraXRes(), signal->SpectraYRes(), 4, img, 0);

        delete[] img;
    }
*/

    *progress = 1;


    for (auto p : processors)
        delete p;

    for (auto b : buffers)
        delete b;

    fft_destroy_plan(cpuPlan);
    delete[] winBuf;
    delete[] ffMap;

    //DispDebug("Shutdown time: %lfs", killTimer.Now());
}

void CDIFReader::Spectra(Birch::PluginIQGetter* getter, SignalCDIF* signal, Birch::Timespan span, Birch::Span<double> freqSpan, double tune, std::vector<FIR*>* filters, WindowFunc* window, unsigned dataSize, unsigned overlap, unsigned fftSize, unsigned res, volatile float *progress, volatile bool *stop) {
    if (working) {
        DispError("CDIFReader (Spectra)", "Cannot start read. Read is already in progress.");
        return;
    }
    
    auto timer = Stopwatch();

    working = true;

    switch (getter->Format()) {
        case Birch::DataFormat::Int8:
            spectra<int8_t>(getter,   signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
        case Birch::DataFormat::Int16:
            spectra<int16_t>(getter,  signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
        case Birch::DataFormat::Int32:
            spectra<int32_t>(getter,  signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
        case Birch::DataFormat::Int64:
            spectra<int64_t>(getter,  signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
        case Birch::DataFormat::Float:
            spectra<float_t>(getter,  signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
        case Birch::DataFormat::Double:
            spectra<double_t>(getter, signal, span, freqSpan, tune, filters, window, dataSize, overlap, fftSize, res, progress, stop);
            break;
    }

    working = false;

    DispInfo("CDIFReader", "Spectra Time: %lfs", timer.Now());
}
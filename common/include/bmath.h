#ifndef __BMATH__
#define __BMATH__

#include "birch.h"

namespace Birch {
// Needs to be moved to a shared library
// This is a temporary solution
struct BMath {
    void (*fft)(Complex<double>*,  Complex<double>*, uint);
    void (*ifft)(Complex<double>*, Complex<double>*, uint);

    void (*firFilter)(Birch::Complex<double>* input, uint size, Birch::Complex<double>* impulse, uint impulseSize);
    Birch::Complex<double>* (*resample)(Birch::Complex<double>* input, unsigned size, double inSampleRate, double outSampleRate, unsigned* outSize);
};
};

#endif
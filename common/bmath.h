#include "include/util.h"

void firFilter(Birch::Complex<double>* input, uint size, Birch::Complex<double>* impulse, uint impulseSize);
Birch::Complex<double>* resample(Birch::Complex<double>* input, unsigned size, double inSampleRate, double outSampleRate, unsigned* outSize);
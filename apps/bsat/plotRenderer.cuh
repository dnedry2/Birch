void renderPointsCUDA(const double* const xVals, const double* const yVals, const bool* const filt, const unsigned* const col, unsigned long count, double xMin, double xMax, double yMin, double yMax, int xRes, int yRes, bool useFilter, bool* renderMask, unsigned* pixels);

void blitCUDA(unsigned* dest_cu, const unsigned* src_cu, unsigned dWidth, unsigned dHeight, unsigned sWidth, unsigned sHeight, unsigned centerX, unsigned centerY, unsigned color);

unsigned* copyToCUDA(const unsigned* data, unsigned length);
void freeCUDA(void* data);
void copyFromCUDA(unsigned* dest, const unsigned* src_cu, unsigned length);
void syncCUDA();

unsigned* rotateCCCUDA(const unsigned* data, int width, int height);
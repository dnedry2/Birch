#include "implot.h"
#include "implot_internal.h"

namespace ImPlot {
    template <typename T> IMPLOT_API  void RasterPlotScatter(const char* label_id, const T* xs, const T* ys, int count, double window, int offset=0, int stride=sizeof(T));
}
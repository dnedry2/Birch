#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"

namespace ImPlot {
    template <typename T> IMPLOT_API  void PolarPlotScatter(const char* label_id, const T* ms, const T* as, int count, int offset=0, int stride=sizeof(T));
}

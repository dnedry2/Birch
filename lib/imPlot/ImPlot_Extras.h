#include "implot.h"
#include "implot_internal.h"

#ifndef _IMPLOT_EXTRAS_
#define _IMPLOT_EXTRAS_

namespace ImPlot {
    //returns true if the plot frame is hovered
    IMPLOT_API bool IsPlotFrameHovered();
    IMPLOT_API bool IsLegendHovered();
    IMPLOT_API void HidePlotQuery();
    IMPLOT_API void SetAllPlotLimitsX(double min, double max);
    IMPLOT_API void PlotImage(const char* label_id, ImTextureID user_texture_id, const ImPlotPoint& pos, ImVec2 size, ImVec2 pixOffset = ImVec2(0, 0));
    IMPLOT_API template <typename T> IMPLOT_API  void PlotScatter(const char* label_id, const T* xs, const T* ys, ImU32* colormap, bool* mask, ImDrawList* dl, int count, int offset=0, int stride=sizeof(T));
    //IMPLOT_API template <typename T> IMPLOT_API  void PlotScatterStatic(const char* label_id, const T* xs, const T* ys, ImU32* colormap, bool* mask, int count, int offset=0, int stride=sizeof(T));
    IMPLOT_API void UpdateNext();

    template <typename T> IMPLOT_API void PlotLine(const char* label_id, const T* xs, const T* ys, ImU32* colormap, bool* mask, int count, int offset=0, int stride=sizeof(T));
    template <typename T> IMPLOT_API void PlotShaded(const char* label_id, const T* xs, const T* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref=0, int offset=0, int stride=sizeof(T));
}

#endif
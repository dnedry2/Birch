#include "rasterPlot.h"
#include "implot_internal.h"
#include "implot.h"

namespace ImPlot {
template <typename Getter>
struct GetterRaster {
    GetterRaster(const Getter& getter, double window) :
        Count(getter.Count),
        Window(window),
        Get(getter)
    { }
    inline ImPlotPoint operator()(int idx) const {
        ImPlotPoint plt = Get(idx);
        int line = plt.x / Window;
        float height = 300;

        plt.x -= line * Window;
        plt.y -= line * height; 

        return plt;
    }
    const Getter& Get;
    const double Window;
    const int Count;
};

template <typename Getter>
inline void RasterPlotScatterEx(const char* label_id, const Getter& getter, double window) {
    GetterRaster<Getter> rGet(getter, window);

    if (BeginItem(label_id, ImPlotCol_MarkerOutline)) {
        if (FitThisFrame()) {
            for (int i = 0; i < rGet.Count; ++i) {
                ImPlotPoint p = rGet(i);
                FitPoint(p);
            }
        }

        const ImPlotNextItemData& s = GetItemData();
        ImDrawList& DrawList = *GetPlotDrawList();

        // render markers
        ImPlotMarker marker = s.Marker == ImPlotMarker_None ? ImPlotMarker_Circle : s.Marker;
        if (marker != ImPlotMarker_None) {
            const ImU32 col_line = ImGui::GetColorU32(s.Colors[ImPlotCol_MarkerOutline]);
            const ImU32 col_fill = ImGui::GetColorU32(s.Colors[ImPlotCol_MarkerFill]);

            switch (GetCurrentScale()) {
                case ImPlotScale_LinLin: RenderMarkers(rGet, TransformerLinLin(), DrawList, marker, s.MarkerSize, s.RenderMarkerLine, col_line, s.MarkerWeight, s.RenderMarkerFill, col_fill); break;
                case ImPlotScale_LinLog: RenderMarkers(rGet, TransformerLinLog(), DrawList, marker, s.MarkerSize, s.RenderMarkerLine, col_line, s.MarkerWeight, s.RenderMarkerFill, col_fill); break;
            }
        }
        EndItem();
    }
}
}
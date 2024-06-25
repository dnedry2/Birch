#include "implot.h"
#include "implot_internal.h"

namespace ImPlot {
// Transforms points from polar to cartesian
struct TransformerPolarLinLin {
    TransformerPolarLinLin() : YAxis(GetCurrentYAxis()) {}
    inline ImVec2 operator()(const ImPlotPoint& plt) const {
        ImPlotContext& gp = *GImPlot;
        ImPlotPoint cart = ImPlotPoint((plt.x - fabs(gp.CurrentPlot->XAxis.Range.Min)) * cos(plt.y - gp.CurrentPlot->YAxis[YAxis].Range.Min), (plt.x - fabs(gp.CurrentPlot->XAxis.Range.Min)) * sin(plt.y - gp.CurrentPlot->YAxis[YAxis].Range.Min));
        return ImVec2( (float)(gp.PixelRange[YAxis].Min.x + gp.Mx * cart.x),
                       (float)(gp.PixelRange[YAxis].Min.y + gp.My[YAxis] * cart.y) );
    }
    const int YAxis;
};

template <typename Getter>
inline void PolarPlotScatterEx(const char* label_id, const Getter& getter) {
    if (BeginItem(label_id, ImPlotCol_MarkerOutline)) {
        if (FitThisFrame()) {
            for (int i = 0; i < getter.Count; ++i) {
                ImPlotPoint p = getter(i);
                FitPoint(p); // this happens in polar cords
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
                case ImPlotScale_LinLin: RenderMarkers(getter, TransformerPolarLinLin(), DrawList, marker, s.MarkerSize, s.RenderMarkerLine, col_line, s.MarkerWeight, s.RenderMarkerFill, col_fill); break;
                case ImPlotScale_LinLog: RenderMarkers(getter, TransformerLinLog(), DrawList, marker, s.MarkerSize, s.RenderMarkerLine, col_line, s.MarkerWeight, s.RenderMarkerFill, col_fill); break;
            }
        }
        EndItem();
    }
}
}
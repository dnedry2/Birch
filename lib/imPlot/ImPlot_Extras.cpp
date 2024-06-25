#include "implot.h"
#include "implot_internal.h"

#include <GL/gl3w.h>

#include "logger.hpp"

namespace ImPlot {
bool IsPlotFrameHovered()
{
    ImPlotContext& gp = *GImPlot;
    IM_ASSERT_USER_ERROR(gp.CurrentPlot != NULL, "IsPlotFrameHovered() needs to be called between BeginPlot() and EndPlot()!");
    return gp.CurrentPlot->FrameRect.Contains(ImGui::GetMousePos());
}

ImPlotPlot* GetInternalPlot()
{
    ImPlotContext& gp = *GImPlot;
    IM_ASSERT_USER_ERROR(gp.CurrentPlot != NULL, "GetInternalPlot() needs to be called between BeginPlot() and EndPlot()!");
    return gp.CurrentPlot;
}

void PlotImage(const char* label_id, ImTextureID user_texture_id, const ImPlotPoint& pos, ImVec2 size, ImVec2 pixOffset)
{
    if (BeginItem(label_id)) {
        ImDrawList& DrawList = *GetPlotDrawList();
        ImVec2 p1 = PlotToPixels(pos.x, pos.y) + pixOffset;
        PushPlotClipRect();
        DrawList.AddImage(user_texture_id, p1, p1 + size);
        PopPlotClipRect();
        EndItem();
    }
}

void HidePlotQuery() {
    ImPlotContext& gp = *GImPlot;
    IM_ASSERT_USER_ERROR(gp.CurrentPlot != NULL, "CancelPlotQuery() needs to be called between BeginPlot() and EndPlot()!");
    gp.CurrentPlot->Queried = false;
}

bool IsLegendHovered() {
    ImPlotContext& gp = *GImPlot;
    IM_ASSERT_USER_ERROR(gp.CurrentPlot != NULL, "IsLegendHovered() needs to be called between BeginPlot() and EndPlot()!");
    return gp.CurrentPlot->FrameHovered && gp.CurrentPlot->LegendHovered;
}

void SetAllPlotLimitsX(double min, double max) {
    ImPlotContext& gp = *GImPlot;

    for (int i = 0; i < gp.Plots.GetSize(); i++) {
        ImPlotPlot& plot = *gp.Plots.GetByIndex(i);
        plot.XAxis.SetMin(min);
        plot.XAxis.SetMax(max);
    }
}

void UpdateNext() {
    GImPlot->NextItemData.Update = true;
}

GLuint genBlankFBO(int length, int height) {
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    (GL_FRAMEBUFFER, fbo);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, length, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

    // do i need this?
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    GLenum bufs[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, bufs);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        DispError("implot_static::genBlankFBO", "%s", "Failed to generate FBO. THIS IS THE END!!!!");
    }

    return fbo;
}

void renderPlot(ImDrawList* dl, GLuint fbo) {
    
}

template <typename T>
struct Max_Idx { static const unsigned int Value; };
template <> const unsigned int Max_Idx<unsigned short>::Value = 65535;
template <> const unsigned int Max_Idx<unsigned int>::Value   = 4294967295;

/// Renders primitive shapes in bulk as efficiently as possible.
template <typename Renderer>
inline void RenderPrimitives(const Renderer& renderer, ImDrawList& DrawList, const ImRect& cull_rect) {
    unsigned int prims        = renderer.Prims;
    unsigned int prims_culled = 0;
    unsigned int idx          = 0;
    const ImVec2 uv = DrawList._Data->TexUvWhitePixel;
    while (prims) {
        // find how many can be reserved up to end of current draw command's limit
        unsigned int cnt = ImMin(prims, (Max_Idx<ImDrawIdx>::Value - DrawList._VtxCurrentIdx) / Renderer::VtxConsumed);
        // make sure at least this many elements can be rendered to avoid situations where at the end of buffer this slow path is not taken all the time
        if (cnt >= ImMin(64u, prims)) {
            if (prims_culled >= cnt)
                prims_culled -= cnt; // reuse previous reservation
            else {
                DrawList.PrimReserve((cnt - prims_culled) * Renderer::IdxConsumed, (cnt - prims_culled) * Renderer::VtxConsumed); // add more elements to previous reservation
                prims_culled = 0;
            }
        }
        else
        {
            if (prims_culled > 0) {
                DrawList.PrimUnreserve(prims_culled * Renderer::IdxConsumed, prims_culled * Renderer::VtxConsumed);
                prims_culled = 0;
            }
            cnt = ImMin(prims, (Max_Idx<ImDrawIdx>::Value - 0/*DrawList._VtxCurrentIdx*/) / Renderer::VtxConsumed);
            DrawList.PrimReserve(cnt * Renderer::IdxConsumed, cnt * Renderer::VtxConsumed); // reserve new draw command
        }
        prims -= cnt;
        for (unsigned int ie = idx + cnt; idx != ie; ++idx) {
            if (!renderer(DrawList, cull_rect, uv, idx))
                prims_culled++;
        }
    }
    if (prims_culled > 0)
        DrawList.PrimUnreserve(prims_culled * Renderer::IdxConsumed, prims_culled * Renderer::VtxConsumed);
}

struct TransformerLinLin {
    TransformerLinLin() : YAxis(GetCurrentYAxis()) {}
    // inline ImVec2 operator()(const ImPlotPoint& plt) const { return (*this)(plt.x, plt.y); }
    inline ImVec2 operator()(const ImPlotPoint& plt) const {
        ImPlotContext& gp = *GImPlot;
        return ImVec2( (float)(gp.PixelRange[YAxis].Min.x + gp.Mx * (plt.x - gp.CurrentPlot->XAxis.Range.Min)),
                       (float)(gp.PixelRange[YAxis].Min.y + gp.My[YAxis] * (plt.y - gp.CurrentPlot->YAxis[YAxis].Range.Min)) );
    }
    const int YAxis;
};

// Transforms points for log x and linear y space
struct TransformerLogLin {
    TransformerLogLin() : YAxis(GetCurrentYAxis()) {}
    inline ImVec2 operator()(const ImPlotPoint& plt) const {
        ImPlotContext& gp = *GImPlot;
        double t = ImLog10(plt.x / gp.CurrentPlot->XAxis.Range.Min) / gp.LogDenX;
        double x = ImLerp(gp.CurrentPlot->XAxis.Range.Min, gp.CurrentPlot->XAxis.Range.Max, (float)t);
        return ImVec2( (float)(gp.PixelRange[YAxis].Min.x + gp.Mx * (x - gp.CurrentPlot->XAxis.Range.Min)),
                       (float)(gp.PixelRange[YAxis].Min.y + gp.My[YAxis] * (plt.y - gp.CurrentPlot->YAxis[YAxis].Range.Min)) );
    }
    const int YAxis;
};

// Transforms points for linear x and log y space
struct TransformerLinLog {
    TransformerLinLog() : YAxis(GetCurrentYAxis()) {}
    inline ImVec2 operator()(const ImPlotPoint& plt) const {
        ImPlotContext& gp = *GImPlot;
        double t = ImLog10(plt.y / gp.CurrentPlot->YAxis[YAxis].Range.Min) / gp.LogDenY[YAxis];
        double y = ImLerp(gp.CurrentPlot->YAxis[YAxis].Range.Min, gp.CurrentPlot->YAxis[YAxis].Range.Max, (float)t);
        return ImVec2( (float)(gp.PixelRange[YAxis].Min.x + gp.Mx * (plt.x - gp.CurrentPlot->XAxis.Range.Min)),
                       (float)(gp.PixelRange[YAxis].Min.y + gp.My[YAxis] * (y - gp.CurrentPlot->YAxis[YAxis].Range.Min)) );
    }
    const int YAxis;
};

// Transforms points for log x and log y space
struct TransformerLogLog {
    TransformerLogLog() : YAxis(GetCurrentYAxis()) {}
    inline ImVec2 operator()(const ImPlotPoint& plt) const {
        ImPlotContext& gp = *GImPlot;
        double t = ImLog10(plt.x / gp.CurrentPlot->XAxis.Range.Min) / gp.LogDenX;
        double x = ImLerp(gp.CurrentPlot->XAxis.Range.Min, gp.CurrentPlot->XAxis.Range.Max, (float)t);
               t = ImLog10(plt.y / gp.CurrentPlot->YAxis[YAxis].Range.Min) / gp.LogDenY[YAxis];
        double y = ImLerp(gp.CurrentPlot->YAxis[YAxis].Range.Min, gp.CurrentPlot->YAxis[YAxis].Range.Max, (float)t);
        return ImVec2( (float)(gp.PixelRange[YAxis].Min.x + gp.Mx * (x - gp.CurrentPlot->XAxis.Range.Min)),
                       (float)(gp.PixelRange[YAxis].Min.y + gp.My[YAxis] * (y - gp.CurrentPlot->YAxis[YAxis].Range.Min)) );
    }
    const int YAxis;
};

// Interprets separate arrays for X and Y points as ImPlotPoints
template <typename T>
struct GetterXsYsx {
    GetterXsYsx(const T* xs, const T* ys, int count, int offset, int stride) :
        Xs(xs),
        Ys(ys),
        Count(count),
        Offset(count ? ImPosMod(offset, count) : 0),
        Stride(stride)
    { }
    inline ImPlotPoint operator()(int idx) const {
        return ImPlotPoint((double)OffsetAndStride(Xs, idx, Count, Offset, Stride), (double)OffsetAndStride(Ys, idx, Count, Offset, Stride));
    }
    const T* const Xs;
    const T* const Ys;
    const int Count;
    const int Offset;
    const int Stride;
};

template <typename Getter, typename Transformer>
inline void RenderLineStripColored(const Getter& getter, const Transformer& transformer, ImDrawList& DrawList, float line_weight, ImU32* col, bool* mask) {
    ImPlotContext& gp = *GImPlot;
    //if (ImHasFlag(gp.CurrentPlot->Flags, ImPlotFlags_AntiAliased) || gp.Style.AntiAliasedLines) {
    ImVec2 p1 = transformer(getter(0));
    if (mask != nullptr) {
        for (int i = 1; i < getter.Count; ++i) {
            ImVec2 p2 = transformer(getter(i));
            if (mask[i] && gp.CurrentPlot->PlotRect.Overlaps(ImRect(ImMin(p1, p2), ImMax(p1, p2))))
                DrawList.AddLine(p1, p2, col[i], line_weight);
            p1 = p2;
        }
    }
    else {
        for (int i = 1; i < getter.Count; ++i) {
            ImVec2 p2 = transformer(getter(i));
            if (gp.CurrentPlot->PlotRect.Overlaps(ImRect(ImMin(p1, p2), ImMax(p1, p2))))
                DrawList.AddLine(p1, p2, col[i], line_weight);
            p1 = p2;
        }
    }
    //}
    //else {
    //    RenderPrimitives(LineStripRenderer<Getter,Transformer>(getter, transformer, col, line_weight), DrawList, gp.CurrentPlot->PlotRect);
    //}
}

#define SQRT_1_2 0.70710678118f
#define SQRT_3_2 0.86602540378f

inline void TransformMarker(ImVec2* points, int n, const ImVec2& c, float s) {
    for (int i = 0; i < n; ++i) {
        points[i].x = c.x + points[i].x * s;
        points[i].y = c.y + points[i].y * s;
    }
}

inline void RenderMarkerGeneral(ImDrawList& DrawList, ImVec2* points, int n, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    TransformMarker(points, n, c, s);
    if (fill)
        DrawList.AddConvexPolyFilled(points, n, col_fill);
    if (outline && !(fill && col_outline == col_fill)) {
        for (int i = 0; i < n; ++i)
            DrawList.AddLine(points[i], points[(i+1)%n], col_outline, weight);
    }
}

inline void RenderMarkerCircle(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[10] = {ImVec2(1.0f, 0.0f),
                         ImVec2(0.809017f, 0.58778524f),
                         ImVec2(0.30901697f, 0.95105654f),
                         ImVec2(-0.30901703f, 0.9510565f),
                         ImVec2(-0.80901706f, 0.5877852f),
                         ImVec2(-1.0f, 0.0f),
                         ImVec2(-0.80901694f, -0.58778536f),
                         ImVec2(-0.3090171f, -0.9510565f),
                         ImVec2(0.30901712f, -0.9510565f),
                         ImVec2(0.80901694f, -0.5877853f)};
    RenderMarkerGeneral(DrawList, marker, 10, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerDiamond(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[4] = {ImVec2(1, 0), ImVec2(0, -1), ImVec2(-1, 0), ImVec2(0, 1)};
    RenderMarkerGeneral(DrawList, marker, 4, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerSquare(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[4] = {ImVec2(SQRT_1_2,SQRT_1_2),ImVec2(SQRT_1_2,-SQRT_1_2),ImVec2(-SQRT_1_2,-SQRT_1_2),ImVec2(-SQRT_1_2,SQRT_1_2)};
    RenderMarkerGeneral(DrawList, marker, 4, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerUp(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[3] = {ImVec2(SQRT_3_2,0.5f),ImVec2(0,-1),ImVec2(-SQRT_3_2,0.5f)};
    RenderMarkerGeneral(DrawList, marker, 3, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerDown(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[3] = {ImVec2(SQRT_3_2,-0.5f),ImVec2(0,1),ImVec2(-SQRT_3_2,-0.5f)};
    RenderMarkerGeneral(DrawList, marker, 3, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerLeft(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[3] = {ImVec2(-1,0), ImVec2(0.5, SQRT_3_2), ImVec2(0.5, -SQRT_3_2)};
    RenderMarkerGeneral(DrawList, marker, 3, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerRight(ImDrawList& DrawList, const ImVec2& c, float s, bool outline, ImU32 col_outline, bool fill, ImU32 col_fill, float weight) {
    ImVec2 marker[3] = {ImVec2(1,0), ImVec2(-0.5, SQRT_3_2), ImVec2(-0.5, -SQRT_3_2)};
    RenderMarkerGeneral(DrawList, marker, 3, c, s, outline, col_outline, fill, col_fill, weight);
}

inline void RenderMarkerAsterisk(ImDrawList& DrawList, const ImVec2& c, float s, bool /*outline*/, ImU32 col_outline, bool /*fill*/, ImU32 /*col_fill*/, float weight) {
    ImVec2 marker[6] = {ImVec2(SQRT_3_2, 0.5f), ImVec2(0, -1), ImVec2(-SQRT_3_2, 0.5f), ImVec2(SQRT_3_2, -0.5f), ImVec2(0, 1),  ImVec2(-SQRT_3_2, -0.5f)};
    TransformMarker(marker, 6, c, s);
    DrawList.AddLine(marker[0], marker[5], col_outline, weight);
    DrawList.AddLine(marker[1], marker[4], col_outline, weight);
    DrawList.AddLine(marker[2], marker[3], col_outline, weight);
}

inline void RenderMarkerPlus(ImDrawList& DrawList, const ImVec2& c, float s, bool /*outline*/, ImU32 col_outline, bool /*fill*/, ImU32 /*col_fill*/, float weight) {
    ImVec2 marker[4] = {ImVec2(1, 0), ImVec2(0, -1), ImVec2(-1, 0), ImVec2(0, 1)};
    TransformMarker(marker, 4, c, s);
    DrawList.AddLine(marker[0], marker[2], col_outline, weight);
    DrawList.AddLine(marker[1], marker[3], col_outline, weight);
}

inline void RenderMarkerCross(ImDrawList& DrawList, const ImVec2& c, float s, bool /*outline*/, ImU32 col_outline, bool /*fill*/, ImU32 /*col_fill*/, float weight) {
    ImVec2 marker[4] = {ImVec2(SQRT_1_2,SQRT_1_2),ImVec2(SQRT_1_2,-SQRT_1_2),ImVec2(-SQRT_1_2,-SQRT_1_2),ImVec2(-SQRT_1_2,SQRT_1_2)};
    TransformMarker(marker, 4, c, s);
    DrawList.AddLine(marker[0], marker[2], col_outline, weight);
    DrawList.AddLine(marker[1], marker[3], col_outline, weight);
}

//!custom rendering function to handle coloring
template <typename Transformer, typename Getter>
inline void RenderMarkersColoredx(Getter getter, Transformer transformer, ImDrawList& DrawList, ImPlotMarker marker, float size, bool rend_mk_line, ImU32* col_mk_line, float weight, bool rend_mk_fill, ImU32* col_mk_fill, bool* mask) {
    static void (*marker_table[ImPlotMarker_COUNT])(ImDrawList&, const ImVec2&, float s, bool, ImU32, bool, ImU32, float) = {
        RenderMarkerCircle,
        RenderMarkerSquare,
        RenderMarkerDiamond,
        RenderMarkerUp,
        RenderMarkerDown,
        RenderMarkerLeft,
        RenderMarkerRight,
        RenderMarkerCross,
        RenderMarkerPlus,
        RenderMarkerAsterisk
    };
    ImPlotContext& gp = *GImPlot;
    if (mask != nullptr) {
        for (int i = 0; i < getter.Count; ++i) {
            if (mask[i]) {
                ImVec2 c = transformer(getter(i));
                if (gp.CurrentPlot->PlotRect.Contains(c))
                    marker_table[marker](DrawList, c, size, rend_mk_line, col_mk_line[i], rend_mk_fill, col_mk_fill[i], weight);
            }
        }
    } else {
        for (int i = 0; i < getter.Count; ++i) {
            ImVec2 c = transformer(getter(i));
            if (gp.CurrentPlot->PlotRect.Contains(c))
                marker_table[marker](DrawList, c, size, rend_mk_line, col_mk_line[i], rend_mk_fill, col_mk_fill[i], weight);
        }
    }
}

template <typename Getter>
inline void PlotLineExxx(const char* label_id, const Getter& getter, ImU32* colormap, bool* mask = nullptr) {
    if (BeginItem(label_id, ImPlotCol_Line)) {
        if (FitThisFrame()) {
            if (mask == nullptr) {
                for (int i = 0; i < getter.Count; ++i) {
                    ImPlotPoint p = getter(i);
                    FitPoint(p);
                }
            } else {
                for (int i = 0; i < getter.Count; ++i) { //!account for mask when fitting
                    if (mask[i]) {
                        ImPlotPoint p = getter(i);
                        FitPoint(p);
                    }
                }
            }
        }
        const ImPlotNextItemData& s = GetItemData();
        ImDrawList& DrawList = *GetPlotDrawList();
        if (getter.Count > 1 && s.RenderLine) {
            //const ImU32 col_line    = ImGui::GetColorU32(s.Colors[ImPlotCol_Line]);
            switch (GetCurrentScale()) {
                case ImPlotScale_LinLin: RenderLineStripColored(getter, TransformerLinLin(), DrawList, s.LineWeight, colormap, mask); break;
                case ImPlotScale_LogLin: RenderLineStripColored(getter, TransformerLogLin(), DrawList, s.LineWeight, colormap, mask); break;
                case ImPlotScale_LinLog: RenderLineStripColored(getter, TransformerLinLog(), DrawList, s.LineWeight, colormap, mask); break;
                case ImPlotScale_LogLog: RenderLineStripColored(getter, TransformerLogLog(), DrawList, s.LineWeight, colormap, mask); break;
            }
        }
        // render markers
        if (s.Marker != ImPlotMarker_None) {
            //const ImU32 col_line = ImGui::GetColorU32(s.Colors[ImPlotCol_MarkerOutline]);
            //const ImU32 col_fill = ImGui::GetColorU32(s.Colors[ImPlotCol_MarkerFill]);

            switch (GetCurrentScale()) {
                case ImPlotScale_LinLin: RenderMarkersColoredx(getter, TransformerLinLin(), DrawList, s.Marker, s.MarkerSize, s.RenderMarkerLine, colormap, s.MarkerWeight, s.RenderMarkerFill, colormap, mask); break;
                case ImPlotScale_LogLin: RenderMarkersColoredx(getter, TransformerLogLin(), DrawList, s.Marker, s.MarkerSize, s.RenderMarkerLine, colormap, s.MarkerWeight, s.RenderMarkerFill, colormap, mask); break;
                case ImPlotScale_LinLog: RenderMarkersColoredx(getter, TransformerLinLog(), DrawList, s.Marker, s.MarkerSize, s.RenderMarkerLine, colormap, s.MarkerWeight, s.RenderMarkerFill, colormap, mask); break;
                case ImPlotScale_LogLog: RenderMarkersColoredx(getter, TransformerLogLog(), DrawList, s.Marker, s.MarkerSize, s.RenderMarkerLine, colormap, s.MarkerWeight, s.RenderMarkerFill, colormap, mask); break;
            }
        }
        EndItem();
    }
}

//!custom version (colormap / filter)
template <typename T>
void PlotLine(const char* label_id, const T* xs, const T* ys, ImU32* colormap, bool* mask, int count, int offset, int stride) {
    GetterXsYsx<T> getter(xs,ys,count,offset,stride);
    return PlotLineExxx(label_id, getter, colormap, mask);
}

template IMPLOT_API void PlotLine<ImS8>(const char* label_id, const ImS8* xs, const ImS8* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImU8>(const char* label_id, const ImU8* xs, const ImU8* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImS16>(const char* label_id, const ImS16* xs, const ImS16* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImU16>(const char* label_id, const ImU16* xs, const ImU16* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImS32>(const char* label_id, const ImS32* xs, const ImS32* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImU32>(const char* label_id, const ImU32* xs, const ImU32* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImS64>(const char* label_id, const ImS64* xs, const ImS64* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<ImU64>(const char* label_id, const ImU64* xs, const ImU64* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<float>(const char* label_id, const float* xs, const float* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);
template IMPLOT_API void PlotLine<double>(const char* label_id, const double* xs, const double* ys, ImU32* colormap, bool* mask, int count, int offset, int stride);

// Accepts colormap
// Also sets opacity to 50
template <typename TGetter1, typename TGetter2, typename TTransformer>
struct ShadedRendererx {
    ShadedRendererx(const TGetter1& getter1, const TGetter2& getter2, const TTransformer& transformer, ImU32* col, bool* mask, float opacity) :
        Getter1(getter1),
        Getter2(getter2),
        Transformer(transformer),
        Prims(ImMin(Getter1.Count, Getter2.Count) - 1),
        Col(col),
        Mask(mask),
        UseMask(mask != nullptr),
        Opacity(opacity)
    {
        P11 = Transformer(Getter1(0));
        P12 = Transformer(Getter2(0));
    }

    inline bool operator()(ImDrawList& DrawList, const ImRect& /*cull_rect*/, const ImVec2& uv, int prim) const {
        const int index = prim + 1;

        ImVec2 P21 = Transformer(Getter1(index));
        ImVec2 P22 = Transformer(Getter2(index));
        
        if (UseMask && !Mask[index]) {
            P21.y = NAN;
            P22.y = NAN;
        }

        ImU32 color = Col[index];
        ((char*)&color)[3] = (255 * Opacity);

        // TODO: Culling
        const int intersect = (P11.y > P12.y && P22.y > P21.y) || (P12.y > P11.y && P21.y > P22.y);
        ImVec2 intersection = Intersection(P11,P21,P12,P22);
        DrawList._VtxWritePtr[0].pos = P11;
        DrawList._VtxWritePtr[0].uv  = uv;
        DrawList._VtxWritePtr[0].col = color;
        DrawList._VtxWritePtr[1].pos = P21;
        DrawList._VtxWritePtr[1].uv  = uv;
        DrawList._VtxWritePtr[1].col = color;
        DrawList._VtxWritePtr[2].pos = intersection;
        DrawList._VtxWritePtr[2].uv  = uv;
        DrawList._VtxWritePtr[2].col = color;
        DrawList._VtxWritePtr[3].pos = P12;
        DrawList._VtxWritePtr[3].uv  = uv;
        DrawList._VtxWritePtr[3].col = color;
        DrawList._VtxWritePtr[4].pos = P22;
        DrawList._VtxWritePtr[4].uv  = uv;
        DrawList._VtxWritePtr[4].col = color;
        DrawList._VtxWritePtr += 5;
        DrawList._IdxWritePtr[0] = (ImDrawIdx)(DrawList._VtxCurrentIdx);
        DrawList._IdxWritePtr[1] = (ImDrawIdx)(DrawList._VtxCurrentIdx + 1 + intersect);
        DrawList._IdxWritePtr[2] = (ImDrawIdx)(DrawList._VtxCurrentIdx + 3);
        DrawList._IdxWritePtr[3] = (ImDrawIdx)(DrawList._VtxCurrentIdx + 1);
        DrawList._IdxWritePtr[4] = (ImDrawIdx)(DrawList._VtxCurrentIdx + 3 - intersect);
        DrawList._IdxWritePtr[5] = (ImDrawIdx)(DrawList._VtxCurrentIdx + 4);
        DrawList._IdxWritePtr += 6;
        DrawList._VtxCurrentIdx += 5;
        P11 = P21;
        P12 = P22;
        return true;
    }
    const TGetter1& Getter1;
    const TGetter2& Getter2;
    const TTransformer& Transformer;
    const int Prims;
    ImU32* const Col;
    bool* const Mask;
    const bool UseMask;
    mutable ImVec2 P11;
    mutable ImVec2 P12;
    static const int IdxConsumed = 6;
    static const int VtxConsumed = 5;
    const float Opacity;
};

// Interprets an array of X points as ImPlotPoints where the Y value is a constant reference value
template <typename T>
struct GetterXsYRefx {
    GetterXsYRefx(const T* xs, double y_ref, int count, int offset, int stride) :
        Xs(xs),
        YRef(y_ref),
        Count(count),
        Offset(count ? ImPosMod(offset, count) : 0),
        Stride(stride)
    { }
    inline ImPlotPoint operator()(int idx) const {
        return ImPlotPoint((double)OffsetAndStride(Xs, idx, Count, Offset, Stride), YRef);
    }
    const T* const Xs;
    const double YRef;
    const int Count;
    const int Offset;
    const int Stride;
};

template <typename T>
void PlotShaded(const char* label_id, const T* xs, const T* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride) {
    GetterXsYsx<T> getter1(xs, ys, count, offset, stride);
    GetterXsYRefx<T> getter2(xs, y_ref, count, offset, stride);
    PlotShadedExx(label_id, getter1, getter2, mask, colormap, opacity);
}

template <typename Getter1, typename Getter2>
inline void PlotShadedExx(const char* label_id, const Getter1& getter1, const Getter2& getter2, bool* mask, ImU32* colormap, float opacity) {
    if (BeginItem(label_id, ImPlotCol_Fill)) {
        if (FitThisFrame()) {
            if (mask != nullptr) {
                for (int i = 0; i < ImMin(getter1.Count, getter2.Count); ++i) {
                    if (mask[i]) {
                        ImPlotPoint p1 = getter1(i);
                        ImPlotPoint p2 = getter2(i);
                        FitPoint(p1);
                        FitPoint(p2);
                    }
                }
            }
            else {
                for (int i = 0; i < ImMin(getter1.Count, getter2.Count); ++i) {
                    ImPlotPoint p1 = getter1(i);
                    ImPlotPoint p2 = getter2(i);
                    FitPoint(p1);
                    FitPoint(p2);
                }
            }
        }
        const ImPlotNextItemData& s = GetItemData();
        ImDrawList & DrawList = *GetPlotDrawList();
        if (s.RenderFill) {
            //ImU32 col = ImGui::GetColorU32(s.Colors[ImPlotCol_Fill]);
            switch (GetCurrentScale()) {
                case ImPlotScale_LinLin: RenderPrimitives(ShadedRendererx<Getter1,Getter2,TransformerLinLin>(getter1,getter2,TransformerLinLin(), colormap, mask, opacity), DrawList, GImPlot->CurrentPlot->PlotRect); break;
                case ImPlotScale_LogLin: RenderPrimitives(ShadedRendererx<Getter1,Getter2,TransformerLogLin>(getter1,getter2,TransformerLogLin(), colormap, mask, opacity), DrawList, GImPlot->CurrentPlot->PlotRect); break;
                case ImPlotScale_LinLog: RenderPrimitives(ShadedRendererx<Getter1,Getter2,TransformerLinLog>(getter1,getter2,TransformerLinLog(), colormap, mask, opacity), DrawList, GImPlot->CurrentPlot->PlotRect); break;
                case ImPlotScale_LogLog: RenderPrimitives(ShadedRendererx<Getter1,Getter2,TransformerLogLog>(getter1,getter2,TransformerLogLog(), colormap, mask, opacity), DrawList, GImPlot->CurrentPlot->PlotRect); break;
            }
        }
        EndItem();
    }
}

template IMPLOT_API void PlotShaded<ImS8>(const char* label_id, const ImS8* xs, const ImS8* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImU8>(const char* label_id, const ImU8* xs, const ImU8* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImS16>(const char* label_id, const ImS16* xs, const ImS16* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImU16>(const char* label_id, const ImU16* xs, const ImU16* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImS32>(const char* label_id, const ImS32* xs, const ImS32* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImU32>(const char* label_id, const ImU32* xs, const ImU32* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImS64>(const char* label_id, const ImS64* xs, const ImS64* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<ImU64>(const char* label_id, const ImU64* xs, const ImU64* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<float>(const char* label_id, const float* xs, const float* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);
template IMPLOT_API void PlotShaded<double>(const char* label_id, const double* xs, const double* ys, ImU32* colormap, float opacity, bool* mask, int count, double y_ref, int offset, int stride);

}
#ifndef __PLOT_FIELD_RENDERER_H__
#define __PLOT_FIELD_RENDERER_H__

#include "implot.h"

#include "plot.hpp"
#include "tool.hpp"
#include "window.hpp"
#include "windowImpls.hpp"

// Builds marker table
void InitPlotFieldRenderers();

class PlotFieldRenderer {
public:
    virtual void RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) = 0;
    virtual bool RenderWidget() = 0;
    virtual const char* Name() = 0;
    
    bool AutoFit();
    void SetAutoFit(bool fit);

    explicit PlotFieldRenderer(PlotField* field);
protected:
    PlotField* field;
    bool autofit = false;
};

class RendererShaded : public PlotFieldRenderer {
public:
    void RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) override;
    bool RenderWidget() override;
    const char* Name() override;

    explicit RendererShaded(PlotField* field);
private:
    int pointThres = 100;
    ImPlotMarker marker = ImPlotMarker_Circle;
    float markerSize = 3.5f;
    float opacity = 0.05f;
};

class RendererLine : public PlotFieldRenderer {
public:
    void RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) override;
    bool RenderWidget() override;
    const char* Name() override;

    explicit RendererLine(PlotField* field);
private:
    int pointThres = 256;
    ImPlotMarker marker = ImPlotMarker_Circle;
    float markerSize = 3.5f;
};

enum class ScatterStyle { Points, Stems, Stairs, Lines };
class RendererScatter : public PlotFieldRenderer {
public:
    void RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) override;
    bool RenderWidget() override;
    const char* Name() override;

    explicit RendererScatter(PlotField* field, bool zmod = true, ScatterStyle style = ScatterStyle::Points);
    ~RendererScatter();
private:
    ScatterStyle style = ScatterStyle::Points;

    uint pointThres = 1024;
    uint marker = 0;
    uint markerSize = 1;

    char lastVer = 0;

    bool shiftDown = false;
    bool lastShiftDown = false;

    bool zmod  = true;
    float color[4] = { 1, 1, 1, 1 };

    bool styleChanged = false;

    // For hover preview
    uint lastMarker = 0;
    bool lastHover  = false;
};

class RendererSpectra : public PlotFieldRenderer {
public:
    void RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) override;
    bool RenderWidget() override;
    const char* Name() override;

    int FFTSize();
    int TexSize();
    int Overlap();
    int DatSize();
    WindowFunc* Window();

    explicit RendererSpectra(PlotField* field);
private:
    int fftSize = 8; //size + 4 so 2048
    int texSize = 2048;
    int overlap = 0;
    int datSize = 2048;
    WindowFunc* window = (*GetWindows())[11]; // Hann window

    const char* sizes = "8\00016\00032\00064\000128\000256\000512\0001024\0002048\0004096\0008192\00016384\00032768\00065536\000131072\000262144\000524288\0001048576\0002097152\0004194304\0008388608\00016777216\00033554432\00067108864\0";

    char lastVer = 0;
};

#endif
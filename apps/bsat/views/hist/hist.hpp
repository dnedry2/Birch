#ifndef _HISTVIEW_
#define _HISTVIEW_

#include "decs.hpp"

#include <vector>
#include "string.h"

#include <GL/gl3w.h>

#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"
#include "ImPlot_Extras.h"

#include "view.hpp"
#include "svg.hpp"
#include "gradient.hpp"
#include "tool.hpp"
#include "widgets.hpp"
#include "plot.hpp"
#include "tooltip.hpp"

#include "defines.h"

void InitHistView(std::vector<Gradient *> *grads, std::vector<Tool *> *tPlugs, int *nID, ImGuiIO *io, volatile float *pBar, const float scale);

class Histogram : public IView {
public:
    const char* Name() override { return "Histogram"; }
    void Render() override;
    bool IsUpdating() override;
    void PushTabStyle() override;
    void PopTabStyle() override;
    void ClosePlot(Plot* plot) override;
    void CloseWidget(Widget* widget) override;
    //void PlotContextMenu() override;
    void Update() override; // Update on view limits change
    ImVec4 GetTabActiveColor() override;

    void SessionAddFileCallback(BFile* file) override {};

    std::vector<HistMeasurement*>* Measurements() { return &measurements; }
    PlotField* Field() { return field; }

    Session* GetSession() { return session; }

    Histogram(Session* session);

private:
    // Updates when time, bin count, or field changes
    void update();

    Session* session;

    const ImVec4 tabColor = ImVec4(.25, .25, 0, 1);
    std::vector<Widget*> sidebar;

    Tool *currentTool;
    std::vector<Tool *> tools;

    ImPlotLimits limits         = ImPlotLimits();
    ImPlotLimits lastLimits     = ImPlotLimits();
    ImPlotInputMap plotControls = ImPlotInputMap();
    ImPlotFlags plotFlags = ImPlotFlags_None;
    bool toolChanged = false; //makes the sidebar scroll to selected tool

    bool isLog = false;

    // idk
    bool plotsHovered = false;
    int  axisCount    = 0;

    enum class BinSelection {
        Count,
        Square,
        Sturges,
        Rice
    };

    class PanTool : public Tool {
    public:
        PanTool();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    };

    class ZoomTool : public Tool {
    public:
        ZoomTool();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    };

    class MeasureTool : public Tool {
    public:
        MeasureTool(ImPlotLimits* limits, Histogram* parent);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_update(Plot* plot) override;
    
    private:
        uint        previewIdx = 0;
        ImPlotPoint previewPos;
        ImPlotPoint lastPreviewPos;
        ImPlotPoint lastMousePos;
        bool        placed = false;

        Histogram* parent;
    };

    class Controls_Widget : public Widget {
    friend class Histogram;

    public:
        Controls_Widget(int ID, GLuint icon, Histogram* parent);
        void Render();

        uint BinCount() const;
        bool HideZero() const;
        bool& Changed();

        PlotSettings* Settings() { return &settings; }

        ~Controls_Widget();

    private:
        Histogram* parent;
        int windowSize = 250;
        BinSelection binSelection = BinSelection::Count;
        uint binCount = 10000;

        bool hideZero = true;

        bool changed = false;

        Tooltip tooltips[4];

        PlotSettings settings;

        uint getBinCount(ulong size);
    };
    class Field_Widget : public Widget {
    public:
        Field_Widget(int ID, GLuint icon, Histogram* parent);
        void Render();
        PlotField* GetSelection() { return selectedField; }

        ~Field_Widget();

    private:
        Histogram* parent;
        int windowSize = 250;
        PlotField* selectedField = nullptr;
        char filter[128] = "";

        std::vector<PlotField*>* fields = nullptr;
    };

    class Stats_Widget : public Widget {
    public:
        Stats_Widget(int ID, GLuint icon, Histogram* parent);
        void Render();

        void Update(const double* data, const bool* mask, ulong size, const Birch::Span<double>& limits);

        ~Stats_Widget();
    private:
        double mean = 0;
        double median = 0;
        double mode = 0;
        double stdDev = 0;
        double skew = 0;
        double kurtosis = 0;
        double variance = 0;
        double sum = 0;
        double count = 0;

        std::string stats = "";

        Histogram* parent;

        int windowSize = 175;
    };

    Controls_Widget* controlsWidget;
    Field_Widget*    fieldWidget;
    Stats_Widget*    statsWidget;

    Plot*      plot    = nullptr;
    PlotField* preview = nullptr;
    PlotField* field   = nullptr;
    char       dataVer = 0;

    Birch::Span<ulong> previewBinLimits;

    double* hist = nullptr; // Current histogram
    ulong   histSize = 0;   // Size of histogram

    std::vector<HistMeasurement*> measurements;

    bool startup = true;

    PlotSettings settings;
};

#endif
/*******************************************************************************
* Plotter view - time based data display
*
* Author: Levi Miller
* Date created: 3/21/2021
*******************************************************************************/

#ifndef _PLOTTER_
#define _PLOTTER_

#include "string.h"
#include <vector>
#include <string>
#include <map>

#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"
#include "ImPlot_Extras.h"

#include "view.hpp"
#include "svg.hpp"
#include "gradient.hpp"
#include "color.hpp"
#include "filter.hpp"
#include "bfile.hpp"
#include "widgets.hpp"

#include "include/birch.h"

#include "defines.h"

void InitPlotterView(std::vector<Gradient *> *grads, std::vector<Tool *> *tPlugs, int *nID, ImGuiIO *io, volatile float *pBar, const float scale, GLuint shader, std::map<std::string, GLuint>* textures);

class Plotter : public IView
{
public:
    const char* Name() override { return "Plotter"; }
    void Render() override;
    bool IsUpdating() override;
    void PushTabStyle() override;
    void PopTabStyle() override;
    ImVec4 GetTabActiveColor() override;

    void Update() override;
    void AnnotateJump();
    void ClosePlot(Plot *plot) override;
    void CloseWidget(Widget *widget) override;

    void SessionAddFileCallback(BFile* file) override;

    Plotter(Session* session);
    ~Plotter();

    void ForceNextReload();
private:
    const ImVec4 tabColor = ImVec4(0, 0.25, .5, 1);
    Session *session;

//sidebar widgets
#pragma region Widgets
    // "Drag plots here" widget
    class CreatePlot_Widget : public Widget {
    public:
        CreatePlot_Widget(int id, Tool** tool, std::vector<Plot*>* plots, Session* session, PlotSettings* settings, PlotSyncSettings* syncSettings);
        void Render();
        void RenderMini(int index);

        ~CreatePlot_Widget();
    private:
        std::vector<Plot*>* plots;
        Tool** tool;
        Session* session;
        PlotSettings* settings;
        PlotSyncSettings* syncSettings;
        GLuint plus;
    };

    // Sidebar widget to list annotations
    class AnnotationDisplay_Widget : public Widget {
    public:
        AnnotationDisplay_Widget(int ID, GLuint icon, Plotter* plotter);
        Annotation* Get_Selection();
        void UpdateShow(bool show) { this->show = show; }
        void Update();
        void Render() override;
        void AddAnnotation(Annotation* annotation);
        int Get_AnnotationCount() { return annotations.size(); }

        ~AnnotationDisplay_Widget();

    private:
        Plotter* plotter;
        const int maxDisp = 6;
        int winHeight = 200;
        std::vector<Annotation*> annotations;
        int currentSelection = -1;
        char** textList = NULL;
        int itemCount = 0;
        bool show;
    };

    // Sidebar widget to list measurements
    class MeasurementDisplay_Widget : public Widget {
    public:
        enum class MeasureMode {
            Screen,
            Search,
            Smart
        };

        MeasurementDisplay_Widget(GLuint icon, std::vector<Measurement*>* measurements, int ID);
        void (*On_Selection)(MeasurementDisplay_Widget* widget);
        Measurement* Get_Selection();
        MeasureMode* Get_Mode() { return &mode; }
        MeasureDisplay* Get_DispMode() { return &dispMode; }
        uint Get_Color() { return ImColor(selectedColor[0], selectedColor[1], selectedColor[2], selectedColor[3]); }
        double XMult() { return unitToMult(dispUnit); }
        void Update(bool display) { show = display; }
        void Render() override;

        ~MeasurementDisplay_Widget();

    private:
        enum Unit {
            s,
            ms,
            us
        };

        double unitToMult(Unit s);

        const int maxDisp = 10;
        int winHeight = 330;
        int currentSelection = -1;
        MeasureMode mode = MeasureMode::Smart;
        Unit dispUnit = s;
        char dispUnitStr[7] = " (s)\0";
        std::vector<Measurement*>* measurements;
        bool show;
        int measCount = -1;

        MeasureDisplay dispMode = MeasureDisplay::None;
        float selectedColor[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
    };

    // Sidebar widget for plot controls
    class PlotControls_Widget : public Widget {
    public:
        PlotControls_Widget(int ID, GLuint icon, Plotter* plotter);
        void Render();
        PlotSettings* Get_Settings() { return settings; }

        ~PlotControls_Widget();

    private:
        PlotSettings* settings;
        Plotter* plotter;
        bool back = false;
    };

    // Sidebar plot field selector
    class Fields_Widget : public Widget {
    public:
        Fields_Widget(int ID, GLuint icon, Tool** tool, std::vector<Plot*>* plots, Session* session, PlotSyncSettings* syncSettings, PlotSettings* settings);
        void Render();
        const PlotField* Minimap();

        bool Closable = false;

        ~Fields_Widget();

    private:
        std::vector<Plot*>* plots;

        Tool** tool;

        char filter[128];
        PlotField* selected = nullptr;
        PlotSettings* settings;
        PlotSyncSettings* syncSettings;
        PlotField* minimap = nullptr;

        Tooltip fieldHoverTip = Tooltip();

        Session* session = nullptr;

        bool settingsOpen = false;
    };

    AnnotationDisplay_Widget  *annotationDisplay;
    PlotControls_Widget       *controlsWidget;
    PlotSettings              *plotSettings;
    PlotSyncSettings          plotSyncSettings;
    MeasurementDisplay_Widget *measureWidget;
    CreatePlot_Widget         *makePlotWidget;
    Colormap_Widget           *colorWidget;
    Filter_Widget             *filterWidget;
    Fields_Widget             *fieldsWidget;
#pragma endregion

//global plot variables
#pragma region PlotVariables
    std::vector<Plot *> plots;
    ImPlotInputMap plotControls = ImPlotInputMap();
    ImPlotFlags plotFlags = ImPlotFlags_None;
    ImPlotLimits limits = ImPlotLimits();
    ImPlotLimits lastLimits = ImPlotLimits();
    ImPlotLimits jumpLimits = ImPlotLimits();
    bool anyPlotHovered = false;
    //ImPlotLimits lastAnalysis;
    ImPlotLimits lastSelection;
    Birch::Timespan currentTime;
    const int viewDelay = 5;
    int axisCount = 0;
    int currentStride = 0;
    int currentViewTimer = 0;
    int currentSelectionTimer = 0;
    bool loaded = false;
    bool jumping = false;
    bool plotsHovered = false;
    bool lastPlotsHovered = false;
    bool fullLoad = false;
    bool lastFullLoad = false;
    bool postDTEZoom = false;
    bool cancelFit = false;
#pragma endregion

//global tool variables
#pragma region ToolVariables
    // TODO: Toolbar should probably be his own widget...
    Tool *currentTool;
    std::vector<Tool *> tools;
    std::vector<std::vector<Tool*>> toolGroups;
    std::vector<uint> toolGroupHoverFrames;

    ColorStack globalColormap;
    FilterStack globalFilters;

    bool     showAnEdit       = false;
    bool     toolChanged      = false; //makes the sidebar scroll to selected tool
    bool     showMeasurePopup = false;
    unsigned tooltipTimer     = 0;
    unsigned tooltipFrames    = 30;
    unsigned colorVersion     = 0;
    unsigned filterVersion    = 0;
#pragma endregion

    std::vector<Widget *> sidebar;
    std::vector<ImPlotLimits> zoomStack;

    bool initialLoad = true;
    bool fitPhase = false;
    bool lastFetching = false;
    bool startup = true;
    volatile float plotRatio = 1;
    bool forceReload = false;
    bool skipZoomHistory = false;

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
    class ZoomToolX : public Tool {
    public:
        ZoomToolX();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    };
    class ZoomToolY : public Tool {
    public:
        ZoomToolY();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    };
    class ZoomToolXY : public Tool {
    public:
        ZoomToolXY();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    };

    class MeasureTool : public Tool {
    public:
        MeasureTool(ImPlotLimits* limits, Session* session, MeasurementDisplay_Widget* widget, Plotter* parent);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_update(Plot* plot) override;
        void renderGraphics(Plot* plot) override;

        ImPlotLimits* const limits;
        Session* const session;
        MeasurementDisplay_Widget* const widget;
        bool mouseHoverPip = false;
        ImPlotFlags* pflags = nullptr;
        char pulseMeasureText[4096] = "I have eaten\nthe plums\nthat were in\nthe icebox\n\nand which\nyou were probably\nsaving\nfor breakfast\n\nForgive me\nthey were delicious\nso sweet\nand so cold";
        Plotter* parent = nullptr;

        bool needSort = false;
        Measurement* sortMeas = nullptr;
        bool needSnap = false;
        Birch::Point<double> snapPoint;
        Measurement* snapMeas = nullptr;

        Birch::Point<double> mouseStart;
        bool boxSelComplete = true;

        ImPlotPoint lastMousePos;
    };
    class AnnotateTool : public Tool {
    public:
        AnnotateTool(ImPlotLimits* limits, AnnotationDisplay_Widget* widget);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_release(Plot* plot) override;

        ImPlotLimits* const limits;
        AnnotationDisplay_Widget* const widget;
    };
    class ColorTool : public Tool {
    public:
        ColorTool(ColorStack* colors, Session* session, Colormap_Widget* widget);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_release(Plot* plot) override;
        void on_middle_click(Plot* plot) override;

        ColorStack* const globalColormap;
        Colormap_Widget* const widget;
        Session* const session;
    };
    class FilterTool : public Tool {
    public:
        FilterTool(FilterStack* stack, Session* session, Plotter* plotter);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_release(Plot* plot) override;
        void on_update(Plot* plot) override;
        void on_middle_click(Plot* plot) override;

        void setLimsNorm(ImPlotInputMap* controls, ImPlotFlags* flags);
        void setLimsBox(ImPlotInputMap* controls, ImPlotFlags* flags);
        void setLimsX(ImPlotInputMap* controls, ImPlotFlags* flags);

        ImPlotInputMap* map;
        ImPlotFlags* flags;

        FilterStack* const stack;
        Session* const session;

        Plotter* plotter = nullptr;
    };
    /*
    class MaskTool : public Tool {
    public:
        MaskTool(FilterStack* stack, Session* session, Plotter* plotter);
        ~MaskTool();

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_release(Plot* plot) override;
        void on_update(Plot* plot) override;
        void on_middle_click(Plot* plot) override;
        
        void uploadMaskTexture();
        void updateMaskTexture();
        void deleteMaskTexture();

        ImPlotInputMap* map;
        ImPlotFlags* flags;

        FilterStack* const stack;
        Session* const session;

        Plotter* plotter = nullptr;

        Mask   mask = Mask(2048, 2048, Birch::Timespan());
        uint*  maskPixels = nullptr;
        GLuint maskTexture = 0;
        uint   radius = 20;
    };
    */
    class TuneTool : public Tool {
    public:
        TuneTool(Plotter* plotter);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_release(Plot* plot) override;
        void on_click(Plot* plot) override;
        void on_update(Plot* plot) override;

        ImPlotPoint start;
        bool inProgress = false;

        ImPlotInputMap* map;
        ImPlotFlags* flags;

        bool dragged = false;

        Plotter* plotter = nullptr;
    };

    class ScreenshotTool : public Tool {
    public:
        ScreenshotTool(const std::vector<Tool*>* tools);

    protected:
        void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
        void on_update(Plot* plot) override;

        std::vector<Plot*>  selected;
        bool   lastShiftDown = false;
        bool   popupOpen     = false;
        bool   lastPopupOpen = false;
        bool   captured      = false;
        bool   captureReady  = false;
        Plot*  clickedPlot   = nullptr;
        int    clickedPlotRC = 0;
        ImVec2 clickedPos    = ImVec2(0, 0);

        uint*  screenshotPixels  = nullptr;
        uint   screenshotWidth   = 0;
        uint   screenshotHeight  = 0;
        GLuint screenshotTexture = 0;

        // Store tools in order to render their plot stuff in the screenshot
        const std::vector<Tool*>* tools = nullptr;
        // Needed to initialize tools
        ImPlotInputMap* controlsPtr = nullptr;
        ImPlotFlags*    flagsPtr    = nullptr;
        // Don't reset screenshot tool if a screenshot is in progress
        bool screenshotInProgress = false;
    };

    void updateView(Birch::Timespan view, char *strideText, char *elCountText, bool live);

    void drawFilters(Plot *plotv);
    void annotPlotHandler(Plot *plot);
    void annotPlotPrint(Plot *plot);

    //void annotateEnd(AnnotationEntry_Widget *widget);

    void colorDragHandler(Plot *plot);
    void filterDragHandler(Plot *plot);

    //void analysisClose(Widget *widgetg);
    //void analysisDragHandler(Plot *plotv);
    //void analysisUpdate(Plot *plotv);

    void measurePlotHandler(Plot *plotv);
};

#endif
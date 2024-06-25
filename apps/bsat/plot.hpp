/********************************
* Rewrite
*
* Author: Levi Miller
* Date created: 1/19/2020
*********************************/

#ifndef _PLOT_H_
#define _PLOT_H_

#include <vector>
#include <string>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "implot_internal.h"
#include <GL/gl3w.h>

#include "tool.hpp"
#include "bfile.hpp"
#include "annotation.hpp"
#include "filter.hpp"
#include "plotRenderer.hpp"
#include "logger.hpp"

#include "include/birch.h"

struct PlotSettings {
    bool   LiveLoad   = false;
    bool   FitPhase   = true;
    int    PlotCount  = 5;
    ImVec4 Background = ImVec4(0.066, 0.066, 0.066, 1);
    ImVec4 GridColor  = ImVec4(0.2, 0.2, 0.2, 1);
    GLuint BackImage  = 0;
    GLuint FrontImage = 0;

    bool   ShowErrors  = true;
    float  ErrorAlpha  = 0.25f;

    bool   ShowMinimap = true;
};

// Represents a field to be plotted
// TODO: PlotField is wretched, needs to be completely rewritten
struct PlotField {
    // TODO: Enum class this
    enum Status {
        Full,
        Preview,
        Shit
    };

    char* Name = nullptr;
    char* Catagory = nullptr;
    double* volatile Data = nullptr;
    double* volatile Timing = nullptr;
    volatile ulong* volatile ElementCount = nullptr;
    volatile bool* volatile Fetching = nullptr;
    volatile double ViewElementCount = 0;
    ImU32* volatile Colormap = nullptr;
    ImU32* volatile ShadedColormap = nullptr;
    bool* volatile FilterMask = nullptr;
    volatile bool* UseFilter = nullptr;
    Status LoadStatus;
    void* File = nullptr;
    bool Tuned = false;
    bool Filtered = false;
    bool NeedsUpdate = false;

    bool LogScale = false;

    Birch::Timespan XLimits = Birch::Timespan(0, 0);
    Birch::Timespan YLimits = Birch::Timespan(0, 0);
    Birch::Timespan XOffset = Birch::Timespan(0, 0);
    Birch::Timespan YOffset = Birch::Timespan(0, 0);

    double MaxValue = 0; // Maximum value in the data field. Only used in spectra

    volatile int XRes = 0;
    volatile int YRes = 0;

    volatile bool ColormapReady = false;

    std::vector<Annotation*> Annotations;

    // Spans of time that will be highlighed in red during plotting (if renderer supports it)
    std::vector<Birch::Span<double>> ErrorRegions;

    volatile char DataVersion = 0;
    void IncVersion();

    PlotFieldRenderer* Renderer;

    int          Ticks = 0;
    double*      TickVals = nullptr;
    const char** TickText = nullptr;

    // TODO: Why the fuck is name stored as a pointer and not copied
    // Update: Well I went and made histogram change it so i guess that works out?
    PlotField(char* name, char* catagory, volatile bool* fetching, double* volatile data, double* volatile time, unsigned long* volatile elCount, PlotFieldRenderer* renderer, ImU32* volatile colormap, bool* volatile mask, void* file, ImU32* volatile shadedColormap = nullptr);
};

// Represents a plot field with some per plot information
struct PlotFieldContainer {
    PlotField* Field;

    bool Disabled = false; // Is field currently disabled
    bool Restore  = false; // Should field be restored after single field mode ends

    GLuint       Texture = 0;
    bool         Rasterized = false;
    ImVec2       TextureSize;
    ImPlotPoint* LastTextureDimensions[2] = { nullptr, nullptr };
    ImPlotPoint  TextureDimensions[2];
    char         LastVersion = 0;

    ImVec2       LastRenderedSize;

    PlotFieldContainer(PlotField* field);
};

// Variables to synchronize multiple plots
struct PlotSyncSettings {
    ImPlotLimits*   limits;
    ImPlotInputMap* controls;
    ImPlotFlags*    flags;
    bool*           anyHovered;
    bool*           anyHoveredLast;
};

// Info badges in top right corner of plot
class PlotBadge {
public:
    void Render(const ImPlotPoint& plotPos, const ImVec2& uv0, const ImVec2& size = ImVec2(16, 16));

    PlotBadge(const std::string& tooltip, GLuint texture);
private:
    std::string tooltipText;
    GLuint      texture;
    Tooltip     tooltip;
};

class Plot {
public:
    void Render(const ImVec2& size, Tool** currentTool, Session* session, void (*ctxMenu)(void* data), void* ctxMenuData);

    // Add field to plot. If field is already in plot nothing will happen
    void AddField(PlotField* field);
    // Remove field from plot. Return true if removed or false if field was not found
    bool RemoveField(const std::string& name);
    bool RemoveField(const PlotField*   field);
    // Returns true if plot contains field
    bool HasField(const std::string& name) const;
    bool HasField(const PlotField* field) const;

    // Tell plot to fit y-axis on next render only
    void FitNext();

    // Is y-axis log scaled
    bool IsLogScale() const;
    // Plot contains no fields and would like to be deleted
    bool IsClosed() const;
    // Return true if plot is currently queried
    bool IsQueried() const;
    // Return true if plot autofits y-axis always
    bool IsAutoFitting() const;
    // Return true if plot legend is hovered
    bool IsLegendHovered() const;

    // Plot rectangle in pixels
    const ImRect& PlotRect() const;
    // Current query
    const ImPlotLimits& Query() const;
    // Current plot limits
    const ImPlotLimits& Limits() const;
    // Current fields in plot
    const std::vector<PlotFieldContainer>& Fields() const;

    // Return field that is currently selected, or nullptr if plot is not in single field mode
    PlotField* SelectedField() const;
    // Return field container that is currently selected, or nullptr if plot is not in single field mode
    PlotFieldContainer* SelectedFieldContainer() const;

    Plot(PlotSettings* settings, PlotSyncSettings* syncSettings);
    ~Plot();
private:
    PlotSettings*     settings     = nullptr;
    PlotSyncSettings* syncSettings = nullptr;

    std::vector<PlotFieldContainer> fields;

    bool fitNext   = false;
    bool logScale  = false;
    bool closed    = false;
    bool queried   = false;
    bool autofit   = false;
    bool legendHov = false;

    ImPlotLimits query;
    ImPlotLimits limits;

    ImPlotPlot* imPlotPlot = nullptr;

    PlotFieldContainer* selectedField = nullptr;
    PlotFieldContainer* hoveredField  = nullptr; // Field that is currently hovered in the legend

    // Render badges / over under indicators
    void renderPlotInfo();
    // Render minimap
    void renderMinimap(PlotField* plotfield, const ImVec2& size);

    // Status variables
    PlotField::Status status = PlotField::Status::Full;
    bool invertedX    = false;
    bool invertedY    = false;
    bool overY        = false;
    bool underY       = false;
    bool lastFetching = false;
    bool tuned        = false;
    bool filtered     = false;
    bool logY         = false;

    // Badges
    PlotBadge* badgeOverY  = nullptr;
    PlotBadge* badgeUnderY = nullptr;
    PlotBadge* badgeGood   = nullptr;
    PlotBadge* badgeBad    = nullptr;
    PlotBadge* badgeWarn   = nullptr;
    PlotBadge* badgeFetch  = nullptr;
    PlotBadge* badgeTune   = nullptr;
    PlotBadge* badgeFilter = nullptr;
    PlotBadge* badgeInvY   = nullptr;
    PlotBadge* badgeInvX   = nullptr;
    PlotBadge* badgeLogY   = nullptr;

    // Hide good badge after a few seconds
    // Time that the plot has been good
    Stopwatch goodTimer = Stopwatch();
    // Seconds to show checkmark for
    const double goodTime = 1.5;
    // Hide the badge
    bool hideGood = false;

    // Amount of time to wait before fetching new data on drag
    const double dragLockTime = 0.1;
    Stopwatch dragLockTimer = Stopwatch();

    // Do fields need to update their data
    bool stale = false;
    ImPlotLimits lastLimits;
    ImRect lastPlotRect;

    // Locks for keyboard shortcuts
    bool lockReverse = false;
    bool lockShift   = false;

    // Set y axis limits on next render
    bool setNextY = false;

    // Max Y value in plot, used for sci notation determination
    double maxY = 0;

    // Control single field mode
    bool singleMode     = false;
    bool lastSingleMode = false;

    // Minimap plot context
    ImPlotContext* mmCtx = nullptr;
};

#endif
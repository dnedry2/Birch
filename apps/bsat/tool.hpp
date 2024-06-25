#ifndef _TOOL_
#define _TOOL_

#include <thread>

#include "decs.hpp"

#include "implot.h"
#include "implot_internal.h"
#include "implot.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "include/birch.h"
#include "tooltip.hpp"

class Tool {
public:
    enum class ToolType { IQPlugin, TBDPlugin, Basic };

    Tool(const char* name, const char* help, bool single, GLuint toolbarIcon, GLuint cursorNormal, GLuint cursorClicked, ImVec2 cursorNormalOffset = ImVec2(0, 0), ImVec2 cursorClickedOffset = ImVec2(0, 0), ImGuiKey hotkey = ImGuiKey_None, const char* group = nullptr);
    ~Tool();

    // Called on tool selection in toolbar
    void Select(ImPlotInputMap* controls, ImPlotFlags* flags);
    // Called when mouse clicked with tool selected
    void MouseClicked(Plot* plot);
    // Called when mouse released with tool selected
    void MouseReleased(Plot* plot);
    // Called when mouse middle clicked with tool selected
    void MouseMiddleClicked(Plot* plot);
    // Called every frame with tool selected
    void PlotUpdate(Plot* plot);
    // Render the tool's cursor at the given position
    void RenderCursor(ImVec2 cursorPos) const;
    // Render graphics on plot only. No tool logic here
    void RenderGraphics(Plot* plot);

    virtual ToolType Type() const { return ToolType::Basic; };

    const char* Name()  const;
    const char* Help()  const;
    const char* Group() const;

    void RenderTooltip();
    
    GLuint Icon()   const;
    GLuint Cursor() const;
    
    ImGuiKey Hotkey()    const;
    bool UseSinglePlot() const;

    Tool (const Tool&) = delete;
    Tool& operator=(const Tool&) = delete;
protected:
    GLuint* currentCursor = nullptr; // The current cursor to render
    ImVec2* currentOffset = nullptr; // The current offset to render the cursor at
    
    GLuint toolbarIcon;
    GLuint cursorNormal;
    ImVec2 cursorNormalOffset;
    GLuint cursorClicked;
    ImVec2 cursorClickedOffset;

    bool single = false; // Single field mode

    ImGuiKey hotkey;

    virtual void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) { };
    virtual void on_click(Plot* plot) { };
    virtual void on_release(Plot* plot) { };
    virtual void on_middle_click(Plot* plot) { };
    virtual void on_update(Plot* plot) { };
    virtual void renderGraphics(Plot* plot) { };

    const char* name;
    const char* help;
    const char* group;

    Tooltip tooltip;
};

class IQProcHost : public Tool {
public:
    IQProcHost(const char* name, const char* help, Birch::PluginIQProcessor* plugin);
    ~IQProcHost();

    ToolType Type() const override;

    bool HasSidebar() const;
    Birch::PluginIQProcessor* Plugin() const;
protected:
    virtual void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    virtual void on_click(Plot* plot) override;
    virtual void on_release(Plot* plot) override;
    virtual void on_middle_click(Plot* plot) override;
    virtual void on_update(Plot* plot) override;
    void getSel(const PlotField* field, ImPlotLimits query);

    ImPlotInputMap* controls;
    ImPlotFlags* flags;

    Birch::PluginIQProcessor* plugin = nullptr;

    std::thread* readThread = nullptr;
};

class TBDProcHost : public Tool {
public:
    TBDProcHost(const char* name, const char* help, Birch::PluginTBDProcessor* plugin);
    ~TBDProcHost();

    ToolType Type() const override;

    bool HasSidebar() const;
    Birch::PluginTBDProcessor* Plugin() const;
protected:
    virtual void on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) override;
    virtual void on_click(Plot* plot) override;
    virtual void on_release(Plot* plot) override;
    virtual void on_middle_click(Plot* plot) override;
    virtual void on_update(Plot* plot) override;
    void getSel(const PlotField* field, ImPlotLimits query);

    ImPlotInputMap* controls;
    ImPlotFlags* flags;

    Birch::PluginTBDProcessor* plugin = nullptr;

    std::thread* readThread = nullptr;
};

// =qfdu$6GFfDxaQ+1DFu&[R5GRE1%]&X.3ae&=9q<]8bHu2HYsD;M57jX1udKh4.C

#endif
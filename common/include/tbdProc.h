#ifndef __BIRCH_TBD_PROC__
#define __BIRCH_TBD_PROC__

#include "util.h"
#include "plugin.h"

class PluginTBDProcessor : public Plugin {
public:
    virtual const char* Name() const = 0; // Return name of plugin

    virtual bool Tool() const = 0;            // Return true if plugin should be shown in the toolbar
    virtual bool Sidebar() const = 0;         // Return true if plugin should be shown in the sidebar
    virtual unsigned ToolbarIcon() const = 0; // Return icon for toolbar
    virtual unsigned CursorIcon() const = 0;  // Return icon for cursor

    virtual void RenderPlot(const char* fieldName) = 0; // Called during plot rendering, can be used to render custom elements on plot
    virtual void RenderSidebar() = 0;                   // Called during sidebar rendering, can be used to render custom elements on sidebar
    virtual bool ShouldRenderSidebar() = 0;             // Called to determine if sidebar should be rendered

    virtual void Process(const std::map<std::string, double*>& data, uint len, Birch::Span<double> xSpan, Birch::Span<double> ySpan) = 0; // Called when data is selected to be processed. x/ySpan are the plot selection limits
    virtual void Init() = 0; // Called when plugin is loaded

    bool _forceMainThread = false; // Internal use only
};

#endif
#ifndef __BIRCH_IQ_PROC__
#define __BIRCH_IQ_PROC__

#include "util.h"
#include "plugin.h"

struct IQBuffer {
    double*  I;              // I samples
    double*  Q;              // Q samples
    double*  AM;             // AM samples
    double*  PM;             // PM samples
    double*  FM;             // FM samples
    double*  TOA;            // TOA of samples
    unsigned ElCount;        // Number of elements in buffer
    double   SampleRate;     // Sample rate of buffer
    double   CenterFreq;     // Center frequency of buffer
};

class PluginIQProcessor : public Plugin {
public:
    virtual const char* Name() const = 0; // Return name of plugin

    virtual bool Tool() const = 0;            // Return true if plugin should be shown in the toolbar
    virtual bool Sidebar() const = 0;         // Return true if plugin should be shown in the sidebar
    virtual unsigned ToolbarIcon() const = 0; // Return icon for toolbar
    virtual unsigned CursorIcon() const = 0;  // Return icon for cursor

    virtual void RenderPlot(const char* fieldName) = 0; // Called during plot rendering, can be used to render custom elements on plot
    virtual void RenderSidebar() = 0;                   // Called during sidebar rendering, can be used to render custom elements on sidebar
    virtual bool ShouldRenderSidebar() = 0;             // Called to determine if sidebar should be rendered
    virtual void RenderWindow() = 0;                    // Called during window rendering, can be used to render custom elements on window
    virtual bool ShouldRenderWindow() = 0;              // Called to determine if window should be rendered

    virtual void Process(const IQBuffer* buf, Birch::Span<double> xSpan, Birch::Span<double> ySpan) = 0; // Called when data is selected to be processed. x/ySpan are the plot selection limits
    virtual void Init() = 0; // Called when plugin is loaded

    bool _forceMainThread = false; // Internal use only
};

#endif
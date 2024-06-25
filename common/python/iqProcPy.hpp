#ifndef __B_PYTHON_IQ_PROC__
#define __B_PYTHON_IQ_PROC__

#include "include/birch.h"
#include "python/python.hpp"

#include <string>

class PythonIQProcessorHost : public Birch::PluginIQProcessor {
public:
    explicit PythonIQProcessorHost(PythonIQProcessor* processor);
    ~PythonIQProcessorHost();

    const char* Name() const override;
    bool Tool() const override;
    bool Sidebar() const override;
    unsigned ToolbarIcon() const override;
    unsigned CursorIcon() const override;

    void RenderPlot(const char* fieldName) override;
    void RenderSidebar() override;
    bool ShouldRenderSidebar() override;
    void RenderWindow() override;
    bool ShouldRenderWindow() override;

    void Process(const Birch::IQBuffer* buf, Birch::Span<double> xSpan, Birch::Span<double> ySpan) override;
    void Init() override;

private:
    PythonIQProcessor* processor;
    uint toolbarIcon;
    uint cursorIcon;

    std::string toolbarSVG;
    std::string cursorSVG;

    std::string name;
};

#endif
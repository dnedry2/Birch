#ifndef __B_PYTHON_TBD_PROC__
#define __B_PYTHON_TBD_PROC__

#include "include/birch.h"
#include "python/python.hpp"

#include <string>
#include <map>

class PythonTBDProcessorHost : public Birch::PluginTBDProcessor {
public:
    explicit PythonTBDProcessorHost(PythonTBDProcessor* processor);
    ~PythonTBDProcessorHost();

    const char* Name() const override;
    bool Tool() const override;
    bool Sidebar() const override;
    unsigned ToolbarIcon() const override;
    unsigned CursorIcon() const override;

    void RenderPlot(const char* fieldName) override;
    void RenderSidebar() override;
    bool ShouldRenderSidebar() override;

    void Process(const std::map<std::string, double*>& data, uint len, Birch::Span<double> xSpan, Birch::Span<double> ySpan) override;
    void Init() override;

private:
    PythonTBDProcessor* processor;
    uint toolbarIcon;
    uint cursorIcon;

    std::string toolbarSVG;
    std::string cursorSVG;

    std::string name;
};

#endif
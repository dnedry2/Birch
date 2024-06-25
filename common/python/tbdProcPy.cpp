#include "tbdProcPy.hpp"

using namespace Birch;

PythonTBDProcessorHost::PythonTBDProcessorHost(PythonTBDProcessor* processor) : processor(processor) {
    name = processor->Name();

    toolbarSVG = processor->ToolbarSVG();
    cursorSVG = processor->CursorSVG();

    _forceMainThread = true;
}

PythonTBDProcessorHost::~PythonTBDProcessorHost() {

}

const char* PythonTBDProcessorHost::Name() const {
    return name.c_str();
}

bool PythonTBDProcessorHost::Tool() const {
    return true;
}
bool PythonTBDProcessorHost::Sidebar() const {
    return false;
}
uint PythonTBDProcessorHost::ToolbarIcon() const {
    return toolbarIcon;
}
uint PythonTBDProcessorHost::CursorIcon() const {
    return cursorIcon;
}

void PythonTBDProcessorHost::RenderPlot(const char* fieldName) { }
void PythonTBDProcessorHost::RenderSidebar() { }
bool PythonTBDProcessorHost::ShouldRenderSidebar() {
    return false;
}

void PythonTBDProcessorHost::Process(const std::map<std::string, double*>& data, uint len, Birch::Span<double> xSpan, Birch::Span<double> ySpan) {
    processor->Process(data, len);
}
void PythonTBDProcessorHost::Init() {
    toolbarIcon = SubmitSVG(toolbarSVG.c_str(), 32, 32);
    cursorIcon  = SubmitSVG(cursorSVG.c_str(),  32, 32);
}
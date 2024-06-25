#include "iqProcPy.hpp"

using namespace Birch;

PythonIQProcessorHost::PythonIQProcessorHost(PythonIQProcessor* processor) : processor(processor) {
    name = processor->Name();

    toolbarSVG = processor->ToolbarSVG();
    cursorSVG = processor->CursorSVG();

    _forceMainThread = true;
}

PythonIQProcessorHost::~PythonIQProcessorHost() {

}

const char* PythonIQProcessorHost::Name() const {
    return name.c_str();
}

bool PythonIQProcessorHost::Tool() const {
    return true;
}
bool PythonIQProcessorHost::Sidebar() const {
    return false;
}
uint PythonIQProcessorHost::ToolbarIcon() const {
    return toolbarIcon;
}
uint PythonIQProcessorHost::CursorIcon() const {
    return cursorIcon;
}

void PythonIQProcessorHost::RenderPlot(const char* fieldName) { }
void PythonIQProcessorHost::RenderSidebar() { }
bool PythonIQProcessorHost::ShouldRenderSidebar() {
    return false;
}

void PythonIQProcessorHost::Process(const IQBuffer* buf, Birch::Span<double> xSpan, Birch::Span<double> ySpan) {
    processor->Process(buf->I, buf->Q, buf->TOA, buf->ElCount);
}
void PythonIQProcessorHost::Init() {
    toolbarIcon = SubmitSVG(toolbarSVG.c_str(), 32, 32);
    cursorIcon  = SubmitSVG(cursorSVG.c_str(),  32, 32);
}

void PythonIQProcessorHost::RenderWindow() { }
bool PythonIQProcessorHost::ShouldRenderWindow() {
    return false;
}
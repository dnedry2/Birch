// Checked 2023-03-03

#include <algorithm>

#include "tool.hpp"
#include "plot.hpp"
#include "logger.hpp"
#include "bfile.hpp"

#include "ImPlot_Extras.h"

#include "defines.h"
#include "mainThread.hpp"

using namespace Birch;

static const ImVec2 cursorSize = ImVec2(16, 16);

Tool::Tool(const char* name, const char* help, bool single, GLuint toolbarIcon, GLuint cursorNormal, GLuint cursorClicked, ImVec2 cursorNormalOffset, ImVec2 cursorClickedOffset, ImGuiKey hotkey, const char* group) {
    this->name = strdup(name);
    this->help = strdup(help);
    
    this->toolbarIcon = toolbarIcon;

    this->cursorNormal        = cursorNormal;
    this->cursorNormalOffset  = cursorNormalOffset;
    this->cursorClicked       = cursorClicked;
    this->cursorClickedOffset = cursorClickedOffset;

    this->hotkey = hotkey;
    this->single = single;

    currentCursor = &this->cursorNormal;
    currentOffset = &this->cursorNormalOffset;

    if (group != nullptr)
        this->group = strdup(group);
    else
        this->group = nullptr;
}
Tool::~Tool() {
    free(const_cast<char*>(name));
    free(const_cast<char*>(help));
    free(const_cast<char*>(group));
}

void Tool::Select(ImPlotInputMap* controls, ImPlotFlags* flags) {
    on_selection(controls, flags);
}

void Tool::MouseClicked(Plot* plot) {
    currentCursor = &cursorClicked;
    currentOffset = &cursorClickedOffset;
    
    on_click(plot);
}

void Tool::MouseReleased(Plot* plot) {
    currentCursor = &cursorNormal;
    currentOffset = &cursorNormalOffset;

    on_release(plot);
}

void Tool::MouseMiddleClicked(Plot* plot) {
    on_middle_click(plot);
}

void Tool::PlotUpdate(Plot* plot) {
    on_update(plot);
}

void Tool::RenderGraphics(Plot* plot) {
    renderGraphics(plot);
}

void Tool::RenderCursor(ImVec2 cursorPos) const {
    auto dl  = ImGui::GetForegroundDrawList(ImGui::GetCurrentWindow());
    auto pos = cursorPos - *currentOffset;

    dl->AddImage(t_ImTexID(*currentCursor), pos, pos + cursorSize);
}

GLuint Tool::Icon() const {
    return toolbarIcon;
}
GLuint Tool::Cursor() const {
    return *currentCursor;
}

ImGuiKey Tool::Hotkey() const {
    return hotkey;
}
bool Tool::UseSinglePlot() const {
    return single;
}

const char* Tool::Name() const {
    return name;
}
const char* Tool::Help() const {
    return help;
}
const char* Tool::Group() const {
    return group;
}
void Tool::RenderTooltip() {
    tooltip.Render([](void* data) {
        Tool *tool = (Tool*)data;

        auto key = std::string(ImGui::GetKeyName(tool->Hotkey()));
        std::transform(key.begin(), key.end(), key.begin(), ::toupper);

        ImGui::Text("%s (%s)", tool->Name(), key.c_str());

        ImGui::Separator();
        ImGui::Text("%s", tool->Help());
    }, (void*)this);
}

#pragma region IQProcHost

IQProcHost::IQProcHost(const char* name, const char* help, PluginIQProcessor* plugin)
            : Tool(name, help, true, plugin->ToolbarIcon(), plugin->CursorIcon(), plugin->CursorIcon(), ImVec2(0, 0), ImVec2(0, 0))
{
    this->plugin = plugin;
}
IQProcHost::~IQProcHost() {
    if (readThread != nullptr && readThread->joinable())
        readThread->join();
    
    delete readThread;
}

static inline bool limitsEqual(ImPlotLimits *a, ImPlotLimits *b)
{
    return a->X.Min == b->X.Min && a->X.Max == b->X.Max && a->Y.Min == b->Y.Min && a->Y.Max == b->Y.Max;
}

// X gate selection
static void setInputMapTime(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_None;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_Shift;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;
    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}
// Box selection
static void setInputMapBox(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}

void IQProcHost::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    setInputMapTime(controls, flags);

    this->controls = controls;
    this->flags = flags;
}
void IQProcHost::on_click(Plot* plot) {
    auto field = plot->SelectedField();

    if (field != nullptr) {
        if (!strncmp(field->Name, "Spectra", 7))
        {
            setInputMapBox(controls, flags);
        }
        else
        {
            setInputMapTime(controls, flags);
        }
    }
}
struct procMainArgs {
    PluginIQProcessor* plugin;
    IQBuffer buffer;
    SignalCDIF* selection;
    Timespan time;
    Span<double> yLims;
};
void procMain(void* args) {
    procMainArgs* a = (procMainArgs*)args;

    a->plugin->Process(&(a->buffer), a->time, a->yLims);

    delete a->selection;
    delete a;
}
void IQProcHost::getSel(const PlotField* field, ImPlotLimits query) {
    Timespan time = { query.X.Min, query.X.Max };

    SignalFile* file = (SignalFile*)field->File;
    SignalCDIF* selection = nullptr;

    if (!strncmp(field->Catagory, "IQ", 2)) {
        selection = file->FetchPortion(time);
    } else if (!strncmp(field->Catagory, "Spectra", 7)) {
        //selection = file->FetchPortion(time);

        Span<double> filter = { query.Y.Min * 1000000, query.Y.Max * 1000000 };
        double tune = filter.Center() * -1000000;

        selection = file->FetchPortion(time, &filter, tune);
    } else {
        DispWarning("IQ Process Host", "%s%s%s%s%s", "Unsupported data type. Type was: \"", field->Catagory, "\\", field->Name, "\"");
    }

    if (selection != nullptr) {
        IQBuffer buffer;
        buffer.I = selection->SI();
        buffer.Q = selection->SQ();
        buffer.AM = selection->Amplitude();
        buffer.PM = selection->Phase();
        buffer.FM = selection->Freq();
        buffer.TOA = selection->TOA();
        buffer.ElCount = *selection->ElementCount();
        buffer.SampleRate = selection->SampleRate();
        buffer.CenterFreq = 5000000; // temp

        auto timer = Stopwatch();

        Span<double> yLims = { query.Y.Min, query.Y.Max };

        if (!plugin->_forceMainThread) {
            plugin->Process(&buffer, time, yLims);

            DispInfo("IQProcHost", "Process Time: %lfs", timer.Now());

            delete selection;
        } else {
            procMainArgs* args = new procMainArgs();
            args->plugin = plugin;
            args->buffer = buffer;
            args->selection = selection;
            args->time = time;
            args->yLims = yLims;

            gMainThreadTasks.push_back(new MainThreadTask(procMain, (void*)args));
        }
    }
}
void IQProcHost::on_release(Plot* plot) {
    if (readThread != nullptr && readThread->joinable())
        readThread->join();

        static ImPlotLimits lastQuery;

        auto field = plot->SelectedField();

        if (plot->IsQueried() && field != nullptr)
        {
            auto query = plot->Query();

            if ((query.X.Min != query.X.Max && query.Y.Min != query.Y.Max) && !limitsEqual(&lastQuery, &query))
            {
                //getSel(field, query);

                readThread = new std::thread(&IQProcHost::getSel, this, field, query);

                ImPlot::HidePlotQuery();
            }

            lastQuery = query;
        }
}
void IQProcHost::on_middle_click(Plot* plot) {

}
void IQProcHost::on_update(Plot* plot) {
    plugin->RenderPlot(plot->SelectedField()->Name);
}

bool IQProcHost::HasSidebar() const {
    return plugin->Sidebar();
}
PluginIQProcessor* IQProcHost::Plugin() const {
    return plugin;
}

Tool::ToolType IQProcHost::Type() const {
    return Tool::ToolType::IQPlugin;
}

#pragma endregion




#pragma region TBDProcHost

TBDProcHost::TBDProcHost(const char* name, const char* help, PluginTBDProcessor* plugin)
            : Tool(name, help, true, plugin->ToolbarIcon(), plugin->CursorIcon(), plugin->CursorIcon(), ImVec2(0, 0), ImVec2(0, 0))
{
    this->plugin = plugin;
}
TBDProcHost::~TBDProcHost() {
    if (readThread != nullptr && readThread->joinable())
        readThread->join();
    
    delete readThread;
}

void TBDProcHost::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    setInputMapTime(controls, flags);

    this->controls = controls;
    this->flags = flags;
}
void TBDProcHost::on_click(Plot* plot) {
    auto field = plot->SelectedField();

    if (field != nullptr) {
        setInputMapTime(controls, flags);
    }
}
struct procMainArgsTBD {
    PluginTBDProcessor* plugin;
    std::map<std::string, double*> buffer;
    uint len;
    Timespan time;
    Span<double> yLims;
};
void procMainTBD(void* args) {
    procMainArgsTBD* a = (procMainArgsTBD*)args;

    a->plugin->Process(a->buffer, a->len, a->time, a->yLims);

    delete a;
}
void TBDProcHost::getSel(const PlotField* field, ImPlotLimits query) {
    Timespan time = { query.X.Min, query.X.Max };

    TBDFile* file = (TBDFile*)field->File;
    
    std::vector<std::string> fields; // This will cause issues because of reference in non main thread tasks
    std::vector<double*> buffers;
    uint count = 0;

    if (!strncmp(field->Catagory, "PDW", 3)) {
        file->FetchPortion(time, &fields, &buffers, &count);
    } else {
        DispWarning("TBD Process Host", "%s%s%s%s%s", "Unsupported data type. Type was: \"", field->Catagory, "\\", field->Name, "\"");
        return;
    }

    auto timer = Stopwatch();

    std::map<std::string, double*> buffer;
    
    for (uint i = 0; i < fields.size(); i++) {
        buffer.insert(std::pair<std::string&, double*>(fields[i], buffers[i]));
    }

    Span<double> yLims = { query.Y.Min, query.Y.Max };

    if (!plugin->_forceMainThread) {
        plugin->Process(buffer, count, time, yLims);

        DispInfo("TBDProcHost", "Process Time: %lfs", timer.Now());
    } else {
        procMainArgsTBD* args = new procMainArgsTBD();
        args->plugin = plugin;
        args->buffer = buffer;
        args->len = count;
        args->time = time;
        args->yLims = yLims;

        gMainThreadTasks.push_back(new MainThreadTask(procMainTBD, (void*)args));
    }
}
void TBDProcHost::on_release(Plot* plot) {
    if (readThread != nullptr && readThread->joinable())
        readThread->join();

        static ImPlotLimits lastQuery;

        auto field = plot->SelectedField();

        if (plot->IsQueried() && field != nullptr)
        {
            auto query = plot->Query();

            if ((query.X.Min != query.X.Max && query.Y.Min != query.Y.Max) && !limitsEqual(&lastQuery, &query))
            {
                readThread = new std::thread(&TBDProcHost::getSel, this, field, query);

                ImPlot::HidePlotQuery();
            }

            lastQuery = query;
        }
}
void TBDProcHost::on_middle_click(Plot* plot) {

}
void TBDProcHost::on_update(Plot* plot) {
    plugin->RenderPlot(plot->SelectedField()->Name);
}

bool TBDProcHost::HasSidebar() const {
    return plugin->Sidebar();
}
PluginTBDProcessor* TBDProcHost::Plugin() const {
    return plugin;
}

Tool::ToolType TBDProcHost::Type() const {
    return Tool::ToolType::TBDPlugin;
}

#pragma endregion
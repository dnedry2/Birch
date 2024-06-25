#include "plotter.hpp"
#include "icons.hpp"
#include "logger.hpp"
#include "bfile.hpp"
#include "shellExec.hpp"

#include <string>
#include <algorithm>
#include <iostream>

using namespace Birch;

int setting_MaxDisplayedElements = 3840 * 2;
bool setting_MeasureShowMultiCursor = true;
bool setting_AskCutPDW = false;

std::vector<Gradient *> *gradients   = nullptr;
//std::vector<IPlugin *>  *plugins     = nullptr;
std::vector<Tool *>    *pluginTools = nullptr;
int iqPlugins  = -1;
int tbdPlugins = -1;

int *nextID = nullptr;

static ImGuiIO *imGuiIO = nullptr;

volatile float *progressBarPercent = nullptr;

#pragma region Textures
static GLuint cursor_tex;
static GLuint zoom_tex;
static GLuint zoomCur_tex;
//static GLuint analyze_tex;
static GLuint measure_tex;
static GLuint annotate_tex;
static GLuint note_tex;
static GLuint preview_tex;
static GLuint full_tex;
static GLuint pip_tex;
static GLuint pipEmpty_tex;
static GLuint x_tex;
static GLuint fm_tex;
static GLuint fft_tex;
static GLuint bad_tex;
static GLuint hand_tex;
static GLuint handClosed_tex;
static GLuint handCursor_tex;
static GLuint color_tex;
static GLuint colorCursor_tex;
static GLuint measurePreview_tex;
static GLuint measurePreview2_tex;
static GLuint measureCursor_tex;
static GLuint windowCollapsed_tex;
static GLuint windowShown_tex;
static GLuint plus_tex;
static GLuint filter_tex;
static GLuint logoBig_tex;
static GLuint visible_tex;
static GLuint hidden_tex;
static GLuint filterCursor_tex;
static GLuint input_tex;
static GLuint settings_tex;
static GLuint fields_tex;
static GLuint file_tex;
static GLuint tune_tex;
static GLuint tuneBadge_tex;
static GLuint dteBadge_tex;
static GLuint fetchingBadge_tex;
static GLuint skull_tex;
static GLuint filterBadge_tex;
static GLuint invY_tex;
static GLuint invX_tex;
static GLuint log_tex;
#pragma endregion

static GLuint crtShader_plotter;

//widget functions
#pragma region WidgetLogic
//called by colormap, filter, and controls widget to refresh after a change
void Plotter::Update()
{
    session->RefreshFields();
}
//called by a sidebar widget when it wants to be closed
void Plotter::CloseWidget(Widget *widget)
{
    int i = 0;
    for (auto w : sidebar)
    {
        if (w->ID == widget->ID)
        {
            delete w;
            sidebar.erase(sidebar.begin() + i);

            return;
        }
        i++;
    }
}
struct plotterCtxMenuData {
    Tool**              currentTool;
    bool*               toolChanged;
    std::vector<Tool*>* tools;
    ImPlotInputMap*     plotControls;
    ImPlotFlags*        plotFlags;
};

static void plotterCtxMenu(void* data)
{
    plotterCtxMenuData* ctxData = (plotterCtxMenuData*)data;

    if (ImGui::BeginMenu("Tools")) {
        for (auto tool : *ctxData->tools)
        {
            ImGui::PushID(tool->Name());
            if (ImGui::Selectable("##tool")) {
                tool->Select(ctxData->plotControls, ctxData->plotFlags);
                *ctxData->currentTool = tool;
                *ctxData->toolChanged = true;
            }
            ImGui::SameLine();
            ImGui::Image(t_ImTexID(tool->Icon()), ImVec2(16, 16));
            ImGui::SameLine();
            ImGui::Text("%s", tool->Name());
            ImGui::PopID();
        }

        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Plugins")) {
        for (auto tool : *pluginTools)
        {
            ImGui::PushID(tool->Name());
            if (ImGui::Selectable("##tool")) {
                tool->Select(ctxData->plotControls, ctxData->plotFlags);
                *ctxData->currentTool = tool;
                *ctxData->toolChanged = true;
            }
            ImGui::SameLine();
            ImGui::Image(t_ImTexID(tool->Icon()), ImVec2(16, 16));
            ImGui::SameLine();
            ImGui::Text("%s", tool->Name());
            ImGui::PopID();
        }

        ImGui::EndMenu();
    }
}
// Called by plots when they want to be closed
void Plotter::ClosePlot(Plot *plot)
{
    int i = 0;
    for (auto cPlot : plots)
    {
        if (cPlot == plot)
        {
            delete cPlot;
            plots.erase(plots.begin() + i);

            return;
        }
        i++;
    }
}
#pragma endregion

// Event handler for annotation selection
void Plotter::AnnotateJump()
{
    Annotation *selection = annotationDisplay->Get_Selection();

    // Center annotation in plot
    double offset = (selection->Bounds[1] - selection->Bounds[0]) / 2;

    jumpLimits.X.Min = selection->Position[0] - offset;
    jumpLimits.X.Max = selection->Position[0] + offset;

    jumping = true;
}

// TODO: Remove this
void InitPlotterView(std::vector<Gradient *> *grads, std::vector<Tool *> *tPlugs, int *nID, ImGuiIO *io, volatile float *pBar, const float scale, GLuint crtShader, std::map<std::string, GLuint>* textures)
{
    cursor_tex          = (*textures)["bicons/IconCursor.svg"];
    zoom_tex            = (*textures)["bicons/IconZoom.svg"];
    zoomCur_tex         = (*textures)["bicons/CursorZoom.svg"];
    measure_tex         = (*textures)["bicons/IconMeasure.svg"];
    annotate_tex        = (*textures)["bicons/IconAnnotate.svg"];
    note_tex            = (*textures)["bicons/DecNote.svg"];
    preview_tex         = (*textures)["bicons/BadgeWarning.svg"];
    full_tex            = (*textures)["bicons/BadgeGood.svg"];
    pip_tex             = (*textures)["bicons/DecPip.svg"];
    pipEmpty_tex        = (*textures)["bicons/DecPipEmpty.svg"];
    bad_tex             = (*textures)["bicons/BadgeBad.svg"];
    hand_tex            = (*textures)["bicons/IconPan.svg"];
    handCursor_tex      = (*textures)["bicons/CursorHand.svg"];
    handClosed_tex      = (*textures)["bicons/CursorFist.svg"];
    color_tex           = (*textures)["bicons/IconColor.svg"];
    colorCursor_tex     = (*textures)["bicons/CursorColor.svg"];
    measurePreview_tex  = (*textures)["bicons/DecPipShadow.svg"];
    measurePreview2_tex = (*textures)["bicons/DecPipShadowLight.svg"];
    measureCursor_tex   = (*textures)["bicons/CursorMeasure.svg"];
    x_tex               = (*textures)["bicons/DecCollapsed.svg"];
    windowCollapsed_tex = (*textures)["bicons/DecCollapsed.svg"];
    windowShown_tex     = (*textures)["bicons/DecShown.svg"];
    plus_tex            = (*textures)["bicons/DecShown.svg"];
    filter_tex          = (*textures)["bicons/IconFilter.svg"];
    visible_tex         = (*textures)["bicons/DecEyeShown.svg"];
    hidden_tex          = (*textures)["bicons/DecEyeHidden.svg"];
    filterCursor_tex    = (*textures)["bicons/CursorFilter.svg"];
    input_tex           = (*textures)["bicons/DecInput.svg"];
    settings_tex        = (*textures)["bicons/IconConfig.svg"];
    fields_tex          = (*textures)["bicons/IconPlot.svg"];
    file_tex            = (*textures)["bicons/IconFile.svg"];
    tune_tex            = (*textures)["bicons/IconTune.svg"];
    tuneBadge_tex       = (*textures)["bicons/BadgeTune.svg"];
    fetchingBadge_tex   = (*textures)["bicons/BadgeFetching.svg"];
    filterBadge_tex     = (*textures)["bicons/BadgeFilter.svg"];
    invY_tex            = (*textures)["bicons/BadgeInvertY.svg"];
    invX_tex            = (*textures)["bicons/BadgeInvertX.svg"];
    log_tex             = (*textures)["bicons/BadgeLog.svg"];

    crtShader_plotter = crtShader;

    gradients          = grads;
    pluginTools        = tPlugs;
    nextID             = nID;
    imGuiIO            = io;
    progressBarPercent = pBar;
}

Plotter::Plotter(Session* session)
{
    controlsWidget = new PlotControls_Widget((*nextID)++, settings_tex, this);
    plotSettings = controlsWidget->Get_Settings();

    plotSyncSettings = { &limits, &plotControls, &plotFlags, &plotsHovered, &lastPlotsHovered };

    //session = new Session(setting_MaxDisplayedElements, progressBarPercent, &globalColormap, &globalFilters);
    this->session = session;

    annotationDisplay = new AnnotationDisplay_Widget((*nextID)++, annotate_tex, this);
    measureWidget     = new MeasurementDisplay_Widget(measure_tex, session->Measurements(), (*nextID)++);
    fieldsWidget      = new Fields_Widget((*nextID)++, fields_tex, &currentTool, &plots, session, &plotSyncSettings, plotSettings);
    colorWidget       = new Colormap_Widget((*nextID)++, color_tex, session, (*gradients)[1], (*gradients)[38], gradients);
    filterWidget      = new Filter_Widget((*nextID)++, filter_tex, session);

    sidebar.push_back(controlsWidget);
    sidebar.push_back(fieldsWidget);
    sidebar.push_back(measureWidget);
    sidebar.push_back(annotationDisplay);
    sidebar.push_back(colorWidget);
    sidebar.push_back(filterWidget);

    tools.push_back(new PanTool());
    tools.push_back(new ZoomTool());
    tools.push_back(new ZoomToolX());
    tools.push_back(new ZoomToolY());
    tools.push_back(new ZoomToolXY());
    tools.push_back(new MeasureTool(&limits, session, measureWidget, this));
    tools.push_back(new AnnotateTool(&limits, annotationDisplay));
    tools.push_back(new ColorTool(session->Colormap(), session, colorWidget));
    tools.push_back(new FilterTool(session->Filters(), session, this));
    tools.push_back(new TuneTool(this));
    tools.push_back(new ScreenshotTool(&tools));

    toolGroups = {
        { tools[0] },                               // Pan
        { tools[1], tools[2], tools[3], tools[4] }, // Zoom, X-Zoom, Y-Zoom, XY-Zoom
        { tools[5] },                               // Measure
        { tools[6] },                               // Annotate
        { tools[7] },                               // Color
        { tools[8] },                               // Filter
        { tools[9] },                               // Tune
        { tools[10] }                               // Screenshot
    };

    toolGroupHoverFrames.resize(toolGroups.size(), 0);

    for (const auto& tool : *pluginTools) {
        const IQProcHost* host = (const IQProcHost*)tool;

        if (host->HasSidebar()) {
            sidebar.push_back(new PluginHost_Widget(host->Plugin(), (*nextID)++));
        }
    }

    makePlotWidget = new CreatePlot_Widget((*nextID)++, &currentTool, &plots, session, plotSettings, &plotSyncSettings);

    currentTool = tools[0];

    InitWidgets(x_tex, windowCollapsed_tex, windowShown_tex, visible_tex, hidden_tex, zoom_tex, input_tex);
    //InitPlots(imGuiIO, &currentTool, bad_tex, preview_tex, full_tex, cursor_tex, hand_tex, handClosed_tex, tuneBadge_tex, dteBadge_tex, fetchingBadge_tex, crtShader_plotter, filterBadge_tex, invY_tex, invX_tex, log_tex);
    InitColorWidgets(visible_tex, hidden_tex);
    InitFilterWidgets(visible_tex, hidden_tex);
    InitAnnotations(note_tex);

    // TODO: Should be a setting
    for (auto& file : *session->Files()) {
        if (file->Type() == FileType::Signal) {
            auto sFile = (SignalFile*)file;

            Plot* am = new Plot(plotSettings, &plotSyncSettings);
            am->AddField(sFile->Field(strdup("AM")));
            plots.push_back(am);
            
            Plot* pm = new Plot(plotSettings, &plotSyncSettings);
            pm->AddField(sFile->Field(strdup("PM")));
            plots.push_back(pm);

            Plot* sp = new Plot(plotSettings, &plotSyncSettings);
            sp->AddField(sFile->Field(strdup("Spectra")));
            plots.push_back(sp);
        }

        sidebar.push_back(new Info_Widget(file_tex, file, session, (*nextID)++));
    }
}

bool Plotter::IsUpdating()
{
    return !loaded || session->IsFetching();
}

void Plotter::SessionAddFileCallback(BFile* file) {
    if (plots.size() == 0 && file->Type() == FileType::Signal) {
        auto sFile = (SignalFile*)file;

        Plot* am = new Plot(plotSettings, &plotSyncSettings);
        am->AddField(sFile->Field(strdup("AM")));
        plots.push_back(am);
        
        Plot* pm = new Plot(plotSettings, &plotSyncSettings);
        pm->AddField(sFile->Field(strdup("PM")));
        plots.push_back(pm);

        Plot* sp = new Plot(plotSettings, &plotSyncSettings);
        sp->AddField(sFile->Field(strdup("Spectra")));
        plots.push_back(sp);
    }

    sidebar.push_back(new Info_Widget(file_tex, file, session, (*nextID)++));
}

// TODO: Move to session
/*
void Plotter::AddFile(char *path, Plugin* loader, std::vector<std::string>& openFields)
{
    //!temp until when settings added
    bool first = session->Files()->size() == 0;
    BFile *file = nullptr;

    if (session->AddFile(path, &file, loader))
    {

        if (loader->Type == PluginType::TBDGetter)
            static_cast<PluginTBDGetter*>(loader)->Update = &jumping;

        //add info widget for last added file
        //! -2 because 2 files are added right now, the 1k and cut pdws
        sidebar.push_back(new Info_Widget(file_tex, (*session->Files())[session->Files()->size() - 1], session, (*nextID)++));

        auto sFile = (*session->Files())[session->Files()->size() - 1];
        auto fields = *sFile->Fields();

        //fieldsWidget->AddFields(fields);

        for (PlotField* field : fields) {
            if (!strncmp(field->Catagory, "Spectra", 7)) {
                colorWidget->AddColor(new ColorizerSpectra(&field->Data, &field->Colormap, &field->ElementCount, file->Filename(), &field->XRes, &field->YRes, &field->ColormapReady, &field->DataVersion, colorWidget->Get_SpectraColor(), gradients));
            }
        }

        if (first && openFields.size() == 0)
        {
            /
            if (loader->Type == PluginType::IQGetter) {
                Plot* am = new Plot(plotSettings, &plotSyncSettings);
                am->AddField(sFile->Field(strdup("AM")));
                plots.push_back(am);
                
                Plot* pm = new Plot(plotSettings, &plotSyncSettings);
                pm->AddField(sFile->Field(strdup("PM")));
                plots.push_back(pm);

                Plot* sp = new Plot(plotSettings, &plotSyncSettings);
                sp->AddField(sFile->Field(strdup("Spectra")));
                plots.push_back(sp);
            }
            / // This causes a crash and I have no idea why
           

            first = false;
        } else if (openFields.size() > 0) {
            for (auto field : openFields) {
                //Plot *plot = new Plot((*nextID)++, &limits, &plotControls, &plotFlags, &plotsHovered, &axisCount, &currentTool, &globalFilters, &globalColormap);
                Plot* plot = new Plot(plotSettings, &plotSyncSettings);

                auto fn = strdup(field.c_str());
                auto plotField = sFile->Field(fn);

                if (plotField == nullptr) {
                    DispWarning("Plotter::AddFile", "Field %s not found in file %s", fn, sFile->Filename());

                    free(fn);
                    continue;
                }

                free(fn);

                plot->AddField(plotField);
                plots.push_back(plot);
            }
        }
    }

    session->SetCurrentZoom(session->TotalTime(), true);
    initialLoad = true;
    loaded = true;
}
*/

void Plotter::Render()
{
    if (plotSettings->BackImage != 0) {
        auto pos = ImGui::GetCursorScreenPos();

        ImGui::Image(t_ImTexID(plotSettings->BackImage), ImVec2(ImGui::GetWindowWidth(), ImGui::GetWindowHeight()), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::SetCursorScreenPos(pos);
    }

    ImGui::Columns(2);

    ImGui::BeginGroup();

    // Count plugins on first run
    if (iqPlugins == -1) {
        iqPlugins  = 0;
        tbdPlugins = 0;

        for (auto tool : *pluginTools) {
            if (tool->Type() == Tool::ToolType::IQPlugin) {
                iqPlugins++;
            } else {
                tbdPlugins++;
            }
        }
    }


    // Toolbar (main tools)
    if (ImGui::BeginChildFrame(1, ImVec2(40, 38.5 * toolGroups.size())))
    {
        ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

        uint groupIdx = 0;
        for (auto group : toolGroups)
        {
            auto tool = group[0];
            bool showArrow = group.size() > 1;

            bool currentToolInGroup = false;
            for (auto t : group) {
                if (currentTool == t) {
                    currentToolInGroup = true;
                    break;
                }
            }

            auto dispTool = currentToolInGroup ? currentTool : tool;

            ImGui::PushStyleColor(ImGuiCol_Button, strcmp(dispTool->Name(), currentTool->Name()) == 0 ? bgColor : ImVec4(0, 0, 0, 0));

            ImGui::BeginGroup();

            if (ImGui::ImageButton(t_ImTexID(dispTool->Icon()), ImVec2(20, 20)))
            {
                dispTool->Select(&plotControls, &plotFlags);
                currentTool = dispTool;
                toolChanged = true;
            }
            if (!imGuiIO->WantTextInput && tool->Hotkey() != GLFW_KEY_ESCAPE && ImGui::IsKeyDown(tool->Hotkey())) {
                tool->Select(&plotControls, &plotFlags);
                currentTool = tool;
                toolChanged = true;
            }

            const ImVec2 pos = ImGui::GetCursorScreenPos();

            // Render an arrow if there are multiple tools in the group
            if (showArrow) {
                auto dl = ImGui::GetWindowDrawList();

                ImVec2 end = ImVec2(pos.x + 28, pos.y - 10);

                dl->AddTriangleFilled(end, end - ImVec2(5, 0), end - ImVec2(0, 5), 0xFFB3B3B3);
            }

            ImGui::EndGroup();

            bool popupOpen = false;

            if (tool->Group() != nullptr ) {
                if (ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left))
                    toolGroupHoverFrames[groupIdx]++;
                else
                    toolGroupHoverFrames[groupIdx] = 0;

                if (ImGui::IsItemHovered() && (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || toolGroupHoverFrames[groupIdx] > 15))
                    ImGui::OpenPopup(tool->Group());

                ImGui::SetNextWindowPos(pos + ImVec2(30, -10), ImGuiCond_Always);
                if (ImGui::BeginPopup(tool->Group())) {
                    for (auto t : group) {
                        ImGui::BeginGroup();
                        ImGui::PushID(t->Name());

                        if (ImGui::Selectable("##tool", false, ImGuiSelectableFlags_SelectOnRelease)) {
                            t->Select(&plotControls, &plotFlags);
                            currentTool = t;
                            toolChanged = true;
                        }
                        ImGui::SameLine();
                        ImGui::Image(t_ImTexID(t->Icon()), ImVec2(16, 16));
                        ImGui::SameLine();
                        ImGui::Text("%s", t->Name());

                        ImGui::PopID();
                        ImGui::EndGroup();

                        t->RenderTooltip();
                    }

                    popupOpen = true;
                    ImGui::EndPopup();
                }
            }

            if (!popupOpen)
                tool->RenderTooltip();
            
            groupIdx++;
        }

        ImGui::PopStyleColor(toolGroups.size());
    }
    ImGui::EndChildFrame();

    //toolbar (iq plugins)
    if (iqPlugins > 0)
    {
        ImVec4 modifier = ImVec4(0.0, 0.1, 0.0, 0);

        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Button) + modifier);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg) + modifier);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive) + modifier);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered) + modifier);

        if (ImGui::BeginChildFrame(2, ImVec2(40, 40 * iqPlugins)))
        {
            ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

            for (auto tool : *pluginTools)
            {
                if (tool->Type() != Tool::ToolType::IQPlugin)
                    continue;

                ImGui::PushStyleColor(ImGuiCol_Button, strcmp(tool->Name(), currentTool->Name()) == 0 ? bgColor : ImVec4(0, 0, 0, 0));

                if (ImGui::ImageButton(t_ImTexID(tool->Icon()), ImVec2(20, 20)) || (!imGuiIO->WantTextInput && tool->Hotkey() != GLFW_KEY_ESCAPE && ImGui::IsKeyDown(tool->Hotkey())))
                {
                    tool->Select(&plotControls, &plotFlags);
                    currentTool = tool;
                    toolChanged = true;
                }
            }

            ImGui::PopStyleColor(iqPlugins);
        }
        ImGui::EndChildFrame();
        ImGui::PopStyleColor(4);
    }

    //toolbar (tbd plugins)
    if (tbdPlugins > 0)
    {
        ImVec4 modifier = ImVec4(0.2, 0.0, 0.0, 0);

        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Button) + modifier);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg) + modifier);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive) + modifier);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered) + modifier);

        if (ImGui::BeginChildFrame(3, ImVec2(40, 40 * tbdPlugins)))
        {
            ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

            for (auto tool : *pluginTools)
            {
                if (tool->Type() != Tool::ToolType::TBDPlugin)
                    continue;

                ImGui::PushStyleColor(ImGuiCol_Button, strcmp(tool->Name(), currentTool->Name()) == 0 ? bgColor : ImVec4(0, 0, 0, 0));

                if (ImGui::ImageButton(t_ImTexID(tool->Icon()), ImVec2(20, 20)) || (!imGuiIO->WantTextInput && tool->Hotkey() != GLFW_KEY_ESCAPE && ImGui::IsKeyDown(tool->Hotkey())))
                {
                    tool->Select(&plotControls, &plotFlags);
                    currentTool = tool;
                    toolChanged = true;
                }
            }

            ImGui::PopStyleColor(tbdPlugins);
        }
        ImGui::EndChildFrame();
        ImGui::PopStyleColor(4);
    }

    ImGui::EndGroup();

    //plots
    ImGui::SameLine();
    if (ImGui::BeginChild("PlotChild")) {
        // Minimap
        // TODO: Extract to class
        const bool showPlots = session->TotalTime().End > 0;
        if (plotSettings->ShowMinimap && showPlots) {
            const float minimapHeight = 64;

            ImPlot::FitNextPlotAxes(false, true);
            auto time = session->TotalTime();
            ImPlot::SetNextPlotLimitsX(time.Start, time.End, ImGuiCond_Always);
            if (ImPlot::BeginPlot("##minimap", NULL, NULL, ImVec2(-1,minimapHeight), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations)) {
                const auto mmField = fieldsWidget->Minimap();
                
                if (mmField != nullptr) {
                    BFile* file = (BFile*)mmField->File;
                    auto mmDispField = file->Minimap(mmField);

                    if (mmDispField != nullptr) {
                        if (file->Type() == FileType::Signal) {
                            ImPlot::PlotLine("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
                        } else {
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1);
                            ImPlot::PlotScatter("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
                            ImPlot::PopStyleVar();
                        }
                    }
                }

                ImPlot::PushPlotClipRect();

                auto dl = ImPlot::GetPlotDrawList();

                const auto& mmLimits = ImPlot::GetPlotLimits().Y;

                {
                    auto tl = ImPlot::PlotToPixels(ImVec2(limits.X.Min, mmLimits.Max));
                    auto br = ImPlot::PlotToPixels(ImVec2(limits.X.Max, mmLimits.Min));

                    dl->AddRectFilled(tl, br, 0x1FFFFFFF, 0);
                }

                if (zoomStack.size() > 0) {
                    auto tl = ImPlot::PlotToPixels(ImVec2(zoomStack.back().X.Min, mmLimits.Max));
                    auto br = ImPlot::PlotToPixels(ImVec2(zoomStack.back().X.Max, mmLimits.Min));

                    dl->AddRectFilled(tl, br, 0x0FFFFFFF, 0);
                }

                ImPlot::PopPlotClipRect();

                bool mmCtrlClicked = false;

                if (ImPlot::DragLineX("##mm_start", &limits.X.Min, false)) {
                    mmCtrlClicked = true;
                    
                    if (limits.X.Min >= limits.X.Max)
                        limits.X.Max = limits.X.Min + 1;
                }
                
                if (ImPlot::DragLineX("##mm_end",   &limits.X.Max, false)) {
                    mmCtrlClicked = true;

                    if (limits.X.Max <= limits.X.Min)
                        limits.X.Min = limits.X.Max - 1;
                }

                Span<double> mmSpan = { limits.X.Min, limits.X.Max };

                if (limits.X.Min < time.Start) {
                    double start = time.Start;
                    if (ImPlot::DragLineX("##mm_start", &start, false)) {
                        limits.X.Min = start;
                        mmSpan.Start = start;
                    }
                }
                if (limits.X.Max > time.End) {
                    double end = time.End;
                    if (ImPlot::DragLineX("##mm_end", &end, false)) {
                        limits.X.Max = end;
                        mmSpan.End   = end;
                    }
                }

                static bool mmLastDrag = false;

                float dragRadius = (ImPlot::PlotToPixels(mmSpan.End, 0).x - ImPlot::PlotToPixels(mmSpan.Start, 0).x) * 0.5;

                double mmLength = mmSpan.Length() * .5;
                double mmCenter = mmSpan.Start + mmLength;
                double yVal     = mmLimits.Max - (mmLimits.Max - mmLimits.Min) * .5;

                if (ImPlot::DragPoint("##center", &mmCenter, &yVal, false, ImVec4(0, 0, 0, 0), dragRadius)) {
                    if (!mmLastDrag)
                        zoomStack.push_back(limits);

                    limits.X.Min = mmCenter - mmLength;
                    limits.X.Max = mmCenter + mmLength;

                    if (limits.X.Min < time.Start)
                        limits.X.Min = time.Start;

                    if (limits.X.Max > time.End)
                        limits.X.Max = time.End;
                    
                    mmCtrlClicked = true;

                    mmLastDrag = true;
                } else if (mmLastDrag) {
                    mmLastDrag = false;

                    jumpLimits = limits;
                    jumping = true;
                }

                if (ImPlot::IsPlotHovered())
                {
                    const auto& io = ImGui::GetIO();
                    ImGui::GetForegroundDrawList(ImGui::GetCurrentWindow())->AddImage(t_ImTexID(cursor_tex), io.MousePos - ImVec2(3, 0), io.MousePos + ImVec2(16, 16) - ImVec2(3, 0), ImVec2(0, 0));

                    if (io.MouseClicked[0] && !mmCtrlClicked && !ImPlot::IsPlotQueried())
                    {
                        const auto plotMousePos = ImPlot::GetPlotMousePos().x;
                        jumpLimits = limits;

                        jumpLimits.X.Min = plotMousePos - mmLength;
                        jumpLimits.X.Max = plotMousePos + mmLength;

                        if (jumpLimits.X.Min < time.Start)
                            jumpLimits.X.Min = time.Start;

                        if (jumpLimits.X.Max > time.End)
                            jumpLimits.X.Max = time.End;

                        jumping = true;

                        zoomStack.push_back(limits);
                    } else if (io.MouseClicked[1] && !mmCtrlClicked && !ImPlot::IsPlotQueried()) {
                        const auto plotMousePos = ImPlot::GetPlotMousePos().x;
                        jumpLimits = limits;

                        if (plotMousePos < jumpLimits.X.Min) {
                            jumpLimits.X.Min = plotMousePos;
                        } else if (plotMousePos > jumpLimits.X.Max) {
                            jumpLimits.X.Max = plotMousePos;
                        } else if (plotMousePos - jumpLimits.X.Min < jumpLimits.X.Max - plotMousePos) {
                            jumpLimits.X.Min = plotMousePos;
                        } else {
                            jumpLimits.X.Max = plotMousePos;
                        }

                        jumping = true;

                        zoomStack.push_back(limits);
                    }
                }

                ImPlot::EndPlot();
            }

            // Paging
            if (!imGuiIO->WantTextInput) {
                static bool pageRC = false;
                if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
                    if (!pageRC) {
                        const auto len = limits.X.Max - limits.X.Min;
                        jumpLimits = limits;

                        jumpLimits.X.Max = jumpLimits.X.Min;
                        jumpLimits.X.Min -= len;

                        jumping = true;
                    }
                    
                    pageRC = true;
                } else if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
                    if (!pageRC) {
                        const auto len = limits.X.Max - limits.X.Min;
                        jumpLimits = limits;

                        jumpLimits.X.Min = jumpLimits.X.Max;
                        jumpLimits.X.Max += len;

                        jumping = true;
                    }

                    pageRC = true;
                } else if (pageRC) {
                    pageRC = false;
                }
            }
        }

        if (ImGui::BeginChild("Plots"))
        {
            // Zoom history
            if (imGuiIO->MouseClicked[ImGuiMouseButton_Middle] && zoomStack.size() > 0)
            {
                const auto& back = zoomStack.back();
                if (limits.X.Min == back.X.Min || limits.X.Max == back.X.Max)
                    zoomStack.pop_back();

                if (zoomStack.size() > 0) {
                    jumping = true;
                    jumpLimits = zoomStack.back();
                    limits = zoomStack.back();

                    zoomStack.pop_back();
                }
            }

            if (showPlots)
            {
                fullLoad = true;

                if (!lastFullLoad)
                    fitPhase = true;

                int plotCount = 1;

                if (plots.size() > 0)
                    plotCount = plots.size() > plotSettings->PlotCount ? plotSettings->PlotCount : plots.size();

                
                float plotSize = ((ImGui::GetContentRegionAvail().y) - 6.5 * plotCount) / plotCount;

                if (initialLoad)
                {
                    limits = ImPlotLimits();
                    limits.X.Min = session->TotalTime().Start;
                    limits.X.Max = session->TotalTime().End;

                    lastLimits = limits;
                    //ImPlot::SetAllPlotLimitsX(limits.X.Min, limits.X.Max);
                    ImPlot::FitNextPlotAxes(false, true);

                    zoomStack.push_back(limits);
                }

                //load new portion of file if zoom has changed
                if (!initialLoad && (lastLimits.X.Min != limits.X.Min || lastLimits.X.Max != limits.X.Max || jumping))
                {
                    loaded = false;
                                                  
                    currentViewTimer = 0;

                    lastLimits = limits;
                }

                if (jumping)
                {
                    limits = jumpLimits;
                    lastLimits = limits;

                    ImPlot::SetAllPlotLimitsX(limits.X.Min, limits.X.Max);
                }

                // load data from disk
                if (forceReload || (loaded == false && (!(imGuiIO->MouseDown[0] | imGuiIO->MouseDown[1]) && currentViewTimer++ >= viewDelay)) || (jumping))
                {
                    currentTime = Timespan(limits.X.Min, limits.X.Max);
                    session->SetCurrentZoom(currentTime, plotSettings->LiveLoad);

                    if (!jumping && ! skipZoomHistory)
                        zoomStack.push_back(lastLimits);

                    loaded = true;
                    jumping = false;
                    fitPhase = true;
                    forceReload = false;
                }

                skipZoomHistory = false;

                bool fetching = session->IsFetching();

                if (initialLoad && fetching)
                    ImPlot::FitNextPlotAxes(false, true);

                plotsHovered = false;

                makePlotWidget->RenderMini(0);

                plotterCtxMenuData ctxData = { &currentTool, &toolChanged, &tools, &plotControls, &plotFlags };

                if (ImPlot::BeginAlignedPlots("Plots"))
                {
                    for (int i = 0; i < plots.size(); i++)
                    {
                        if (initialLoad)
                            ImPlot::FitNextPlotAxes(false, true);
                        
                        if (cancelFit) {
                            imGuiIO->MouseDoubleClicked[0] = false;
                            imGuiIO->MouseClickedCount[0]  = 0;
                            imGuiIO->MouseClickedTime[0]   = 0;

                            ImPlot::FitNextPlotAxes(false, false, false, false);
                        }

                        //plots[i]->Render(ImVec2(-1, plotSize), 1, fullLoad, &plotsHovered, lastPlotsHovered, currentTool->UseSinglePlot(), plotSettings, plotterCtxMenu, &ctxData);
                        plots[i]->Render(ImVec2(-1, plotSize), &currentTool, session, plotterCtxMenu, &ctxData);

                        if (plots[i]->IsClosed())
                            ClosePlot(plots[i]);

                        makePlotWidget->RenderMini(i + 1);
                    }

                    ImPlot::EndAlignedPlots();
                }

                cancelFit = false;

                if (plots.size() == 0)
                    makePlotWidget->Render();
                
                if (initialLoad)
                {
                    lastLimits = limits;

                    if (!fetching)
                        initialLoad = false;
                }

                if (!fetching && lastFetching)
                    fitPhase = true;

                lastPlotsHovered = plotsHovered;

                lastFetching = fetching;
                lastFullLoad = fullLoad;

                if (session->NeedsUpdate()) {
                    forceReload = true;
                    skipZoomHistory = true;
                }
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    ImGui::NextColumn();

    if (startup)
        ImGui::SetColumnOffset(ImGui::GetColumnIndex(), ImGui::GetWindowSize().x - 240);

    // Not only will you die on that hill, the hill will die too

    // Sidebar
    ImGui::PushID(this);
    if (ImGui::BeginChild("Tools"))
    {
        //display sidebar widgets
        ImGuiWindowClass windowClass;
        windowClass.ClassId = ImGui::GetID(this);
        windowClass.DockingAllowUnclassed = false;

        ImGuiID dockspace_id = ImGui::GetID(this);
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode, &windowClass);

        for (Widget* w : sidebar) {
            ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_Once);
            ImGui::SetNextWindowClass(&windowClass);

            if (toolChanged && w->ToolID != nullptr && !strncmp(w->ToolID, currentTool->Name(), 32)) {
                ImGui::SetNextWindowFocus();
                toolChanged = false;
            }

            w->Render();
        }

        if (colorWidget->Version() != colorVersion) {
            colorVersion = colorWidget->Version();
            Update();
        }
        if (filterWidget->Version() != filterVersion) {
            filterVersion = filterWidget->Version();
            Update();
        }
    }
    ImGui::EndChild();
    ImGui::PopID();

    ImGui::EndColumns();

    startup = false;
}

void Plotter::ForceNextReload()
{
    forceReload = true;
}

const static ImVec4 multVec(const ImVec4 v, const float s) {
    return ImVec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

void Plotter::PushTabStyle() {
    ImGui::PushStyleColor(ImGuiCol_TabHovered, multVec(tabColor, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Tab, tabColor);
    ImGui::PushStyleColor(ImGuiCol_TabActive, multVec(tabColor, 1.5f));
}

void Plotter::PopTabStyle() {
    ImGui::PopStyleColor(3);
}

ImVec4 Plotter::GetTabActiveColor() {
    return multVec(tabColor, 1.5f);
}

Plotter::~Plotter()
{
    for (Widget* w : sidebar)
        delete w;

    for (Plot* p : plots)
        delete p;

    delete makePlotWidget;
    delete plotSettings;
}
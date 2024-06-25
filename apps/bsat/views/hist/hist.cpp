#include "hist.hpp"
#include "hist.cuh"
#include "logger.hpp"

#include "icons.hpp"


static volatile bool* forceUpdate = nullptr;

static std::vector<Gradient *> *gradients   = nullptr;
static std::vector<Tool *>     *pluginTools = nullptr;

static int *nextID = nullptr;

static ImGuiIO *imGuiIO = nullptr;

static volatile float *progressBarPercent = nullptr;

#pragma region Textures
static GLuint cursor_tex;
static GLuint zoom_tex;
static GLuint zoomCur_tex;
static GLuint analyze_tex;
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
static GLuint skull_tex;
#pragma endregion


constexpr static uint previewSize = 4096;

static double* histogramProcessor(const double* data, const bool* mask, ullong dataSize, uint bins, uint previewSize, Birch::Span<double> limits) {
    auto timer = Stopwatch();

    double* output = new double[previewSize]; // Minimized histogram for rendering
    uint* hist     = new uint[bins];          // Full histogram for processing

    // Init buffers to zero
    std::fill_n(output, previewSize, 0);
    std::fill_n(hist,   bins,        0);

    // Find min/max
    double dataMin = data[0];
    double dataMax = data[0];

    #pragma omp parallel for reduction(min: dataMin) reduction(max: dataMax)
    for (ullong i = 0; i < dataSize; i++) {
        const double val = data[i];

        if (mask[i]) {
            dataMin = std::min(dataMin, val);
            dataMax = std::max(dataMax, val);
        }
    }

    // Calculate histogram
    const double binSize = (dataMax - dataMin) / bins;

    #pragma omp parallel for
    for (ullong i = 0; i < dataSize; i++) {
        const double val = data[i];

        if (mask[i]) {
            const uint index = (uint)((val - dataMin) / binSize);

            #pragma omp atomic
            hist[index]++;
        }
    }

    // Scale histogram to preview size
    const double scale = (limits.End - limits.Start) / (double)previewSize;

    for (uint i = limits.Start; i < limits.End && i < bins; i++) {
        const uint index = (uint)(i / scale);

        output[index] = std::max((double)hist[i], output[index]);
    }

    delete[] hist;

    DispInfo("HistProcessor", "Histogram time: %lfs", timer.Now());

    return output;
}

static double* buildHist(const double* data, const bool* mask, ulong dataLen, uint bins, double min, double max) {
    double* hist = new double[bins];

    // Init buffers to zero
    std::fill_n(hist, bins, 0);

    // Calculate histogram
    const double binSize = (max - min) / bins;

    #pragma omp parallel for
    for (ulong i = 0; i < dataLen; i++) {
        const double val = data[i];

        if (mask[i]) {
            const uint index = (uint)((val - min) / binSize);

            #pragma omp atomic
            hist[index]++;
        }
    }

    return hist;
}

const static ImVec4 multVec(const ImVec4 v, const float s) {
    return ImVec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

struct histStats {
    ulong  count    = 0;
    double sum      = 0;
    double mean     = 0;
    double median   = 0;
    double mode     = 0;
    double variance = 0;
    double stdDev   = 0;
    double skew     = 0;
    double kurtosis = 0;
};

histStats calcStats(const double* data, const bool* mask, ulong size, const Birch::Span<double>& limits) {
    histStats out;

    // Calc sum and count
    for (ulong i = (ulong)limits.Start; i < limits.End; i++) {
        if (mask[i]) {
            out.count++;
            out.sum += data[i];
        }
    }

    out.mean = out.sum / out.count;

    // I'll do the rest later...

    return out;
}

void InitHistView(std::vector<Gradient *> *grads, std::vector<Tool *> *tPlugs, int *nID, ImGuiIO *io, volatile float *pBar, const float scale)
{
    cursor_tex = textures["bicons/IconCursor.svg"];
    zoom_tex = textures["bicons/IconZoom.svg"];
    zoomCur_tex = textures["bicons/IconZoomCursor.svg"];
    analyze_tex = textures["bicons/IconAnalyze.svg"];
    measure_tex = textures["bicons/IconMeasure.svg"];
    annotate_tex = textures["bicons/IconAnnotate.svg"];
    note_tex = textures["bicons/IconNote.svg"];
    preview_tex = textures["bicons/IconPreview.svg"];
    full_tex = textures["bicons/IconFull.svg"];
    pip_tex = textures["bicons/IconMeasurePip.svg"];
    pipEmpty_tex = textures["bicons/IconMeasurePipEmpty.svg"];
    fm_tex = textures["bicons/IconFM.svg"];
    fft_tex = textures["bicons/IconFFT.svg"];
    bad_tex = textures["bicons/IconBad.svg"];
    hand_tex = textures["bicons/IconHand.svg"];
    handCursor_tex = textures["bicons/IconHandCursor.svg"];
    handClosed_tex = textures["bicons/IconHandClosed.svg"];
    color_tex = textures["bicons/IconColor.svg"];
    colorCursor_tex = textures["bicons/IconColorCursor.svg"];
    measurePreview_tex = textures["bicons/IconMeasurePipPreview.svg"];
    measurePreview2_tex = textures["bicons/IconMeasurePipPreview2.svg"];
    measureCursor_tex = textures["bicons/IconMeasureCursor.svg"];
    x_tex = textures["bicons/IconX.svg"];
    windowCollapsed_tex = textures["bicons/IconWindowCollapsed.svg"];
    windowShown_tex = textures["bicons/IconWindowExpanded.svg"];
    plus_tex = textures["bicons/IconWindowExpanded.svg"];
    filter_tex = textures["bicons/IconFilter.svg"];
    visible_tex = textures["bicons/IconVisible.svg"];
    hidden_tex = textures["bicons/IconHidden.svg"];
    filterCursor_tex = textures["bicons/IconFilterCursor.svg"];
    input_tex = textures["bicons/IconInput.svg"];
    settings_tex = textures["bicons/IconSettings.svg"];
    fields_tex = textures["bicons/IconFields.svg"];
    file_tex = textures["bicons/IconFile.svg"];
    tune_tex = textures["bicons/IconTuneTool.svg"];
    tuneBadge_tex = textures["bicons/IconTuneBadge.svg"];
    dteBadge_tex = textures["bicons/IconDTEBadge.svg"];
    skull_tex = textures["bicons/IconSkull.svg"];
    gradients = grads;
    pluginTools = tPlugs;
    nextID = nID;
    imGuiIO = io;
    progressBarPercent = pBar;
}

Histogram::Histogram(Session* session) {
    this->session = session;

    controlsWidget = new Controls_Widget((*nextID)++, settings_tex, this);
    fieldWidget    = new Field_Widget((*nextID)++, fields_tex, this);
    statsWidget    = new Stats_Widget((*nextID)++, measure_tex, this);

    sidebar.push_back(controlsWidget);
    sidebar.push_back(fieldWidget);
    sidebar.push_back(statsWidget);

    tools.push_back(new PanTool());
    tools.push_back(new ZoomTool());
    tools.push_back(new MeasureTool(&limits, this));

    currentTool = tools[0];

    //plot = new Plot((*nextID)++, &limits, &plotControls, &plotFlags, &plotsHovered, &axisCount, &currentTool, session->Filters(), session->Colormap());
    plot = new Plot(controlsWidget->Settings(), nullptr);

    preview = new PlotField(strdup(""), strdup(""), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    preview->Renderer = new RendererScatter(preview, false, ScatterStyle::Stems);
}

void Histogram::Render() {
    ImGui::PushID(this);

    // Calculate new preview data if the field selection has changed
    bool needsUpdate = false;

    needsUpdate |= field != fieldWidget->GetSelection();
    needsUpdate |= controlsWidget->Changed();
    needsUpdate |= isLog != plot->IsLogScale();

    isLog = plot->IsLogScale();

    controlsWidget->Changed() = false;
    
    if (field != nullptr) {
        needsUpdate |= field->DataVersion != dataVer;
        dataVer = field->DataVersion;
    }

    if (needsUpdate)
        Update();

    preview->LogScale = isLog;

    // Update stats and y limits
    if (limits.X.Min != lastLimits.X.Min || limits.X.Max != lastLimits.X.Max) {
        int offset = limits.X.Min;

        if (offset < 0)
            offset = 0;
        if (offset >= controlsWidget->BinCount())
            goto updateEnd;

        int count = limits.X.Max;
        if (count > controlsWidget->BinCount())
            count = controlsWidget->BinCount();
        
        count -= offset;

        statsWidget->Update(field->Data, field->FilterMask, *field->ElementCount, preview->XLimits);
        
        double max = *std::max_element(hist + offset, hist + offset + count);
        
        preview->YLimits.End = max;
    }
    updateEnd:

    ImGui::Columns(2);

    ImGui::BeginGroup();

    //toolbar (main tools)
    if (ImGui::BeginChildFrame(1, ImVec2(40, 39 * tools.size())))
    {
        ImVec4 bgColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

        for (auto tool : tools)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, strcmp(tool->Name(), currentTool->Name()) == 0 ? bgColor : ImVec4(0, 0, 0, 0));

            if (ImGui::ImageButton(t_ImTexID(tool->Icon()), ImVec2(20, 20)) || (!imGuiIO->WantTextInput && tool->Hotkey() != GLFW_KEY_ESCAPE && ImGui::IsKeyDown(tool->Hotkey())))
            {
                tool->Select(&plotControls, &plotFlags);
                currentTool = tool;
                toolChanged = true;
            }
        }

        ImGui::PopStyleColor(tools.size());
    }
    ImGui::EndChildFrame();
    ImGui::EndGroup();

    ImGui::SameLine();

    if (ImGui::BeginChild("Plot")) {
       lastLimits = limits;

       plot->Render(ImVec2(-1, -1), &currentTool, session, nullptr, nullptr);
    }
    ImGui::EndChild();

    ImGui::NextColumn();

    if (startup) {
        ImGui::SetColumnOffset(ImGui::GetColumnIndex(), ImGui::GetWindowSize().x - 240);
        startup = false;
    }

    

    //sidebar
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
    }
    ImGui::EndChild();

    ImGui::PopID();
}

bool Histogram::IsUpdating() {
    return false;
}

void Histogram::PushTabStyle() {
    ImGui::PushStyleColor(ImGuiCol_TabHovered, multVec(tabColor, 2.6f));
    ImGui::PushStyleColor(ImGuiCol_Tab, tabColor);
    ImGui::PushStyleColor(ImGuiCol_TabActive, multVec(tabColor, 2.0f));
}

void Histogram::PopTabStyle() {
    ImGui::PopStyleColor(3);
}

void Histogram::ClosePlot(Plot* plot)
{
}

void Histogram::CloseWidget(Widget* widget)
{
}

//void Histogram::PlotContextMenu()
//{
//}

void Histogram::Update()
{
    char* prevField = nullptr;

    if (preview != nullptr) {
        prevField = strdup(preview->Name);
        plot->RemoveField(preview->Name);
    }

    field = fieldWidget->GetSelection();

    preview->Name = field->Name;

    delete[] preview->Data; // Deletes hist
    delete[] preview->Timing;
    delete[] preview->Colormap;
    delete[] preview->FilterMask;


    double max = 0;
    double min = 0;

    #pragma omp parallel for reduction(max:max) reduction(min:min)
    for (ulong i = 0; i < *field->ElementCount; i++) {
        max = std::max(max, field->Data[i]);
        min = std::min(min, field->Data[i]);
    }

    const auto binCount = controlsWidget->BinCount();

    Stopwatch timer;

    #ifndef FFT_IMPL_CUDA
        hist = buildHist(field->Data, field->FilterMask, *field->ElementCount, binCount, min, max);
        DispInfo("HistProcessor (CPU)", "Build time: %lfs", timer.Now());
    #else
        hist = buildHistCUDA(field->Data, field->FilterMask, *field->ElementCount, binCount, min, max);
        DispInfo("HistProcessor (GPU)", "Build time: %lfs", timer.Now());
    #endif

    preview->Colormap   = new unsigned[binCount];
    preview->FilterMask = new bool[binCount];

    std::fill_n(preview->Colormap,   binCount, 0xFFFFFFFF);
    std::fill_n(preview->FilterMask, binCount, true);

    // Set preview's values to new field
    preview->Fetching = field->Fetching;
    preview->File     = field->File;
    preview->Data     = hist;

    preview->Timing = new double[binCount];
    const double step = (max - min) / binCount;

    #pragma omp parallel for
    for (int i = 0; i < binCount; i++)
        preview->Timing[i] = min + i * step;

    if (controlsWidget->HideZero()) {
        #pragma omp parallel for
        for (uint i = 0; i < binCount; i++) {
            preview->FilterMask[i] = hist[i] != 0;
        }
    }

    if (isLog) {
        #pragma omp parallel for
        for (uint i = 0; i < binCount; i++) {
            hist[i] = log(hist[i]);
        }
    }

    preview->XLimits.Start = min;
    preview->XLimits.End   = max;
    preview->YLimits.Start = 0;
    preview->YLimits.End   = *std::max_element(hist, hist + binCount);
    preview->LoadStatus    = PlotField::Status::Full;

    histSize = binCount;
    preview->ElementCount = &histSize;

    if (prevField != nullptr) {
        // Field changed, reset limits
        if (strcmp(prevField, preview->Name)) {
            limits.X.Min = preview->XLimits.Start;
            limits.X.Max = preview->XLimits.End;
            limits.Y.Min = preview->YLimits.Start;
            limits.Y.Max = preview->YLimits.End;
        }

        free(prevField);
    }

    statsWidget->Update(field->Data, field->FilterMask, *field->ElementCount, preview->XLimits);

    plot->AddField(preview);
    plot->FitNext();
}

ImVec4 Histogram::GetTabActiveColor() {
    return multVec(tabColor, 2.0f);
}

static double calcMean(const double* data, const bool* mask, uint size, Birch::Span<double> bounds) {
    double sum = 0;
    uint count = 0;

    for (uint i = 0; i < size; i++) {
        if (mask[i] && bounds.Contains(data[i])) {
            sum += data[i];
            count++;
        }
    }

    return sum / count;
}


Histogram::PanTool::PanTool()   : Tool("Pan",  "", false, hand_tex,  handCursor_tex, handClosed_tex, ImVec2(8, 8), ImVec2(8, 8), ImGuiKey_F) { }
Histogram::ZoomTool::ZoomTool() : Tool("Zoom", "", false, zoom_tex,  zoomCur_tex,    zoomCur_tex,    ImVec2(5, 5), ImVec2(5, 5), ImGuiKey_D) { }

Histogram::MeasureTool::MeasureTool(ImPlotLimits* limits, Histogram* parent)
                       : Tool("Measure", "", false, measure_tex, measure_tex, measure_tex, ImVec2(7, 16), ImVec2(7, 16), ImGuiKey_A), parent(parent) { }

void Histogram::PanTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;
    controls->PanButton = ImGuiMouseButton_Left;

    *flags = ImPlotFlags_None;
}
void Histogram::ZoomTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_None;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Left;
    controls->PanButton = ImGuiMouseButton_Right;

    controls->BoxSelectCancelButton = ImGuiMouseButton_Right;

    *flags = ImPlotFlags_Crosshairs;
}
void Histogram::MeasureTool::on_selection(ImPlotInputMap *controls, ImPlotFlags *flags)
{
    controls->VerticalMod = -1;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_Shift;

    controls->QueryButton = -1;
    controls->QueryMod = -1;

    controls->BoxSelectButton = -1;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}

// Finds the nearest value in a sorted array
static inline uint findNearest(const double* input, uint len, double needle) {
    auto it = std::lower_bound(input, input + len, needle);
    return std::distance(input, std::min(it, input + len - 1));
}

// Finds the nearest value in an unsorted array
static inline uint findNearestUnsorted(const double* input, uint len, double needle) {
    auto it = std::min_element(input, input + len, [needle](double a, double b) {
        return std::abs(a - needle) < std::abs(b - needle);
    });

    return std::distance(input, it);
}

static void plotMeasurement(HistMeasurement* measurement, const double* x, const double* y, const bool* mask, uint count) {
    ImGui::PushID(measurement);
    ImPlot::PushPlotClipRect();
    const auto plotLimits = ImPlot::GetPlotLimits();

    // Fill box
    ImPlot::GetPlotDrawList()->AddRectFilled(
        ImPlot::PlotToPixels(measurement->Bounds()->Start, plotLimits.Y.Max),
        ImPlot::PlotToPixels(measurement->Bounds()->End,   plotLimits.Y.Min),
        IM_COL32(255, 255, 255, 10));

    // Draw edges
    bool update = false;

    update |= ImPlot::DragLineX("##start", &measurement->Bounds()->Start, false);
    update |= ImPlot::DragLineX("##end",   &measurement->Bounds()->End,   false);

    if (update) {
        auto bounds = *measurement->Bounds();

        double mean = calcMean(y, mask, count, bounds);

        if (measurement->Stats() != nullptr)
            free(measurement->Stats());

        measurement->Stats() = strdup(std::to_string(mean).c_str());
    }

    // Draw textbox
    const auto centerPos = ImPlot::PlotToPixels(measurement->Bounds()->Center(), plotLimits.Y.Max - plotLimits.Y.Max * 0.1);
    const auto textSize  = ImGui::CalcTextSize(measurement->Stats());

    // Box
    ImPlot::GetPlotDrawList()->AddRectFilled(
        ImVec2(centerPos.x - textSize.x / 2 - 5, centerPos.y - textSize.y / 2 - 5),
        ImVec2(centerPos.x + textSize.x / 2 + 5, centerPos.y + textSize.y / 2 + 5),
        IM_COL32(0, 0, 0, 128), 5);
    ImPlot::GetPlotDrawList()->AddRect(
        ImVec2(centerPos.x - textSize.x / 2 - 5, centerPos.y - textSize.y / 2 - 5),
        ImVec2(centerPos.x + textSize.x / 2 + 5, centerPos.y + textSize.y / 2 + 5),
        IM_COL32(255, 255, 255, 255), 5);
    
    // Text
    ImPlot::GetPlotDrawList()->AddText(
        ImVec2(centerPos.x - textSize.x / 2, centerPos.y - textSize.y / 2),
        IM_COL32(255, 255, 255, 255),
        measurement->Stats());

    ImPlot::PopPlotClipRect();
    ImGui::PopID();
}

// Find distribution edges from centerpoint
static Birch::Span<double> findCurve(const double* x, const double* y, const bool* mask, ulong size, ulong center, double threshold) {
    Birch::Span<double> curve = Birch::Span<double>(0, 0);

    // Find left edge
    for (ulong i = center; i > 0; i--) {
        if (y[i] < threshold && mask[i]) {
            curve.Start = x[i];
            break;
        }
    }

    // Find right edge
    for (ulong i = center; i < size; i++) {
        if (y[i] < threshold && mask[i]) {
            curve.End = x[i];
            break;
        }
    }

    return curve;
}


void Histogram::MeasureTool::on_update(Plot* plot) {
    static uint pip_tex             = textures["bicons/DecPip.svg"];
    static uint measurePreview_tex  = textures["bicons/DecPipShadow.svg"];
    static uint cursor_tex          = textures["bicons/IconCursor.svg"];

    //static GLuint pip_tex            = svgTextures["bicons/DecPip.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //static GLuint measurePreview_tex = svgTextures["bicons/DecPipShadow.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //static GLuint cursor_tex         = svgTextures["bicons/IconCursor.svg"].GetTexture(ImVec2(16, 16))->TextureID();

    static auto* imGuiIO = &ImGui::GetIO();

    ImPlotPoint mousePos = ImPlot::GetPlotMousePos();

    const double *y      = plot->SelectedField()->Data;
    const double *x      = plot->SelectedField()->Timing;
    const bool   *mask   = plot->SelectedField()->FilterMask;
    const auto    count  = *plot->SelectedField()->ElementCount;

    const double* dataX  = parent->Field()->Timing;
    const double* dataY  = parent->Field()->Data;
    const bool*   dataM  = parent->Field()->FilterMask;
    const auto    dataC  = *parent->Field()->ElementCount;

    auto measurements = parent->Measurements();

    for (auto m : *measurements) {
        plotMeasurement(m, dataX, dataY, dataM, dataC);
    }

    if (ImPlot::IsPlotHovered()) {
        if (lastMousePos.x != mousePos.x || lastMousePos.y != mousePos.y) {
            auto plotLimits = plot->Limits();

            previewIdx = SmartSearch(x, y, mask, count, Birch::Span<double>(plotLimits.Y.Min, plotLimits.Y.Max), mousePos);
            previewPos = ImPlotPoint(x[previewIdx], y[previewIdx]);
        }

        if (lastPreviewPos.x != previewPos.x || lastPreviewPos.y != previewPos.y)
            placed = false;

        if (imGuiIO->MouseReleased[0] && !placed) {
            Birch::Span<double> bounds = findCurve(x, y, mask, count, previewIdx, previewPos.y * 0.1);

            auto m = new HistMeasurement(bounds, plot->SelectedField());

            double mean = calcMean(dataY, dataM, dataC, bounds);

            if (m->Stats() != nullptr)
                free(m->Stats());

            m->Stats() = strdup(std::to_string(mean).c_str());

            measurements->push_back(m);
            placed = true;
        }

        ImPlot::PlotImage("##measure", t_ImTexID(measurePreview_tex), previewPos, ImVec2(16, 16), ImVec2(-7, -16));
    }

    lastMousePos   = mousePos;
    lastPreviewPos = previewPos;
}

Histogram::Controls_Widget::Controls_Widget(int ID, GLuint icon, Histogram* parent) : Widget(ID, icon, std::string("Plot Controls")) {
    Closable = true;
    Collapsed = false;

    this->parent = parent;
}
Histogram::Controls_Widget::~Controls_Widget() {}
void Histogram::Controls_Widget::Render() {
    if (beginWidget(this)) {
        if (parent->field == nullptr) {
            ImGui::Text("No field selected.");
            endWidget();
            return;
        }

        static const char* binSelStrs[]  = { "Count", "Square Root", "Sturges", "Rice" };
        static const char* binSelDescs[] = { u8"Manual input", u8"k = ⌈√n⌉", u8"k = ⌈log₂n⌉ + 1", u8"k = ⌈2∛n⌉" };

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("Bin Selection: ");
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
        if (ImGui::BeginCombo("##binning", binSelStrs[(int)binSelection])) {
            for (int i = 0; i < 4; i++) {
                bool selected = (int)binSelection == i;

                if (ImGui::Selectable(binSelStrs[i], selected)) {
                    if (selected)
                        continue;

                    binSelection = (BinSelection)i;

                    // If set to count, just keep the last number
                    if (binSelection != BinSelection::Count)
                        binCount = getBinCount(*parent->field->ElementCount);
                    
                    changed = true;
                }

                tooltips[i].Render(binSelDescs[i]);
            }

            ImGui::EndCombo();
        }

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("Bin Count: ");
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);

        ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;
        if (binSelection != BinSelection::Count)
            flags |= ImGuiInputTextFlags_ReadOnly;

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
        changed |= ImGui::InputScalar("##binCount", ImGuiDataType_U32, &binCount, nullptr, nullptr, nullptr, flags);

        changed |= ImGui::Checkbox("Hide Zero", &hideZero);
    }
    endWidget();
}
uint Histogram::Controls_Widget::getBinCount(ulong size) {
    switch (binSelection) {
        case BinSelection::Square:
            return (uint)ceil(sqrt(size));
        case BinSelection::Sturges:
            return (uint)ceil(log2(size)) + 1;
        case BinSelection::Rice:
            return (uint)ceil(2 * pow(size, 1.0 / 3.0));
        default:
            return 0;
    }
}
uint Histogram::Controls_Widget::BinCount() const {
    return binCount;
}
bool Histogram::Controls_Widget::HideZero() const {
    return hideZero;
}
bool& Histogram::Controls_Widget::Changed() {
    return changed;
}

Histogram::Field_Widget::Field_Widget(int ID, GLuint icon, Histogram* parent) : Widget(ID, icon, std::string("Fields")) {
    Closable = true;
    Collapsed = false;

    this->parent = parent;
}
Histogram::Field_Widget::~Field_Widget() {}
void Histogram::Field_Widget::Render() {
    if (beginWidget(this)) {
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
        ImGui::InputTextWithHint("##filterFields2", "Search", filter, 128, ImGuiInputTextFlags_AutoSelectAll);
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 28);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);
        ImGui::Image(t_ImTexID(zoom_tex), ImVec2(14, 14));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);

        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg) * ImVec4(0.25, 0.25, 0.25, 1));
        if (ImGui::BeginChildFrame(1234134124, ImVec2(0, 0)))
        {
            ImGui::PopStyleColor();

            for (auto f : parent->GetSession()->Fields())
            {
                // Only show tbd fields
                if (!(!strncmp(f->Catagory, "PDW", 3) && strcasestr(f->Name, filter) != NULL))
                    continue;
                
                ImGui::Selectable(f->Name, selectedField == f, ImGuiSelectableFlags_SpanAvailWidth);

                if (ImGui::IsItemClicked())
                    selectedField = f;
            }
        }
        else
        {
            ImGui::PopStyleColor();
        }
        ImGui::EndChildFrame();
    }
    endWidget();
}

Histogram::Stats_Widget::Stats_Widget(int ID, GLuint icon, Histogram* parent) : Widget(ID, icon, std::string("Stats")) {
    Closable = true;
    Collapsed = false;

    this->parent = parent;
}
Histogram::Stats_Widget::~Stats_Widget() {}
void Histogram::Stats_Widget::Update(const double* data, const bool* mask, ulong size, const Birch::Span<double>& limits) {
    if (data == nullptr || mask == nullptr || size == 0)
        return;

    histStats hStats = calcStats(data, mask, size, limits);

    printf("%lf->%lf\n", limits.Start, limits.End);

    count    = hStats.count;
    sum      = hStats.sum;
    mean     = hStats.mean;
    median   = hStats.median;
    mode     = hStats.mode;
    variance = hStats.variance;
    stdDev   = hStats.stdDev;
    skew     = hStats.skew;
    kurtosis = hStats.kurtosis;

    // Build string output
    stats = "";

    stats += "Count: "              + std::to_string(count)    + "\n";
    stats += "Mean: "               + std::to_string(mean)     + "\n";
    stats += "Median: "             + std::to_string(median)   + "\n";
    stats += "Mode: "               + std::to_string(mode)     + "\n";
    stats += "Standard Deviation: " + std::to_string(stdDev)   + "\n";
    stats += "Variance: "           + std::to_string(variance) + "\n";
    stats += "Skew: "               + std::to_string(skew)     + "\n";
    stats += "Kurtosis: "           + std::to_string(kurtosis);
}

void Histogram::Stats_Widget::Render() {
    if (beginWidget(this)) {
        // Text will be readonly, so this should be fine
        char* cstr = const_cast<char*>(stats.c_str());
        ImGui::InputTextMultiline("##stats", cstr, stats.size() + 1, ImVec2(-1, -1), ImGuiInputTextFlags_ReadOnly);
    }
    endWidget();
}
/*******************************************************************************
* Birch Signal Analysis Tool
*
* 11/10/2020
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <list>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <map>
#include <omp.h>

#define GLEW_STATIC
#define IMGUI_IMPL_OPENGL_LOADER_GLEW

#ifdef _WIN64
#include <Windows.h>

#define strdup _strdup
#endif

#ifdef __unix__
#include <unistd.h>
#endif

#include "ImPlot_Extras.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include "nfd.h"

#include "shader.hpp"

#include "annotation.hpp"
#include "color.hpp"
//#include "dpi.hpp"
#include "filter.hpp"
#include "font.h"
#include "font_bold.h"
#include "gradients.hpp"
#include "icons.hpp"
#include "bfile.hpp"
#include "session.hpp"
#include "widgets.hpp"
#include "svg.hpp"

//#include "missControl.hpp"
#include "plotter.hpp"
//#include "raster.hpp"
#include "hist.hpp"
//#include "polar.hpp"
#include "ambig.hpp"
//#include "live.hpp"

#include "plotRenderer.hpp"

#include "logger.hpp"
#include "stopwatch.hpp"
#include "window.hpp"
#include "windowImpls.hpp"
#include "fft.h"
#include "bmath.h"
#include "include/bmath.h"

#include "project.hpp"
#include "packfile.hpp"
#include "config.hpp"

#ifndef NO_PYTHON
#include "python/python.hpp"
#include "python/iqProcPy.hpp"
#include "python/tbdProcPy.hpp"
#endif

#include "cudaFuncs.cuh"

#include "license.h"

#include "defines.h"

#include "mainThread.hpp"

#include "include/birch.h"

#define PLUGIN_CLIENT
#include "include/plugin.h"

#define BSAT_VERSION "0.11b"

//UI styling settings
#pragma region Styling
const ImVec2 framePad       = ImVec2(5.0f, 5.0f);
const ImVec2 itemSpace      = ImVec2(8.0f, 8.0f);
const ImVec2 itemSpaceSmall = ImVec2(2.0f, 2.0f);
const float  toolbarWidthIn = 1;
const float  iconScale      = 1;
ImVec4       clearColor     = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);
ImColor      menuBack;
#pragma endregion

// All signals will be previewed with this many points
unsigned signalMaxSize = 20000;

// UI textures
std::map<std::string, uint> textures;
std::map<std::string, std::string> markers;

// Shaders
std::map<std::string, Shader*> gShaders;

ImFont* font_normal = nullptr;
ImFont* font_bold   = nullptr;

static std::vector<Gradient *> gradients;
std::vector<Gradient *>* gGradients = &gradients;
static uint fps = 0;

static Config config;
Config& gConfig = config;

static Config theme;

static ullong totalMem = 0;

static std::string computeString = "";

//global SDL, ImGui, and other UI related variables
#pragma region WindowVariables
GLFWwindow *window = nullptr;

ImGuiIO      *imGuiIO;
ImGuiContext *context;

// TODO: This should be an external library
static Birch::BMath bmath = { &fft, &ifft, &firFilter, &resample };

static volatile float progressBarPercent = 0;
volatile float* gProgressBar = &progressBarPercent;

std::vector<MainThreadTask*> gMainThreadTasks = std::vector<MainThreadTask*>();

static   int  nextID  = 32; // TODO: Remove this, should be using PushID / PopID
volatile int* gNextID = &nextID;

#pragma endregion

static std::vector<PluginInterface> plugins;
static std::vector<Tool *>          pluginTools;

std::vector<PluginInterface>* gPlugins = &plugins;

#ifndef NO_PYTHON
static std::vector<PythonIQProcessor*>  pythonIQPlugins;
static std::vector<PythonTBDProcessor*> pythonTBDPlugins;
#endif


struct recentFile {
    std::string path;
    std::string filename;
    std::string pluginName;

    recentFile(const std::string& path, const std::string& filename, const std::string& pluginName)
    : path(path), filename(filename), pluginName(pluginName) { }

    recentFile(const recentFile& other) : path(other.path), filename(other.filename), pluginName(other.pluginName) { }
};

static void saveRecentFiles(const std::vector<recentFile>& files) {
    FILE* file = fopen("recent.lst", "wb");

    if (file == nullptr) {
        DispError("Birch", "Failed to open recent.lst for writing");
        return;
    }

    for (const auto& rf : files) {
        fprintf(file, "%s, ", rf.path.c_str());
        fprintf(file, "%s, ", rf.filename.c_str());
        fprintf(file, "%s\n", rf.pluginName.c_str());
    }

    fclose(file);
}
static void loadRecentFiles(std::vector<recentFile>& files) {
    files.clear();

    FILE* file = fopen("recent.lst", "r");

    if (file == nullptr) {
        DispWarning("Birch", "Failed to open recent.lst for reading");
        return;
    }

    fseek(file, 0, SEEK_END);
    auto size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buf = new char[size + 1];
    fread(buf, 1, size, file);
    buf[size] = '\0';

    fclose(file);

    char path[512];
    char filename[512];
    char pluginName[512];

    char* pos = buf;

    while(sscanf(pos, "%[^,], %[^,], %[^\n]\n", path, filename, pluginName) == 3) {
        pos = strchr(pos, '\n') + 1;
        files.push_back(recentFile(path, filename, pluginName));
    }
}
static void addRecentFile(std::vector<recentFile>& files, const std::string& path, const std::string& filename, const std::string& pluginName) {
    for (auto it = files.begin(); it != files.end(); it++) {
        if (it->path == path) {
            files.erase(it);
            break;
        }
    }

    files.insert(files.begin(), recentFile(path, filename, pluginName));

    if (files.size() > 10)
        files.pop_back();

    saveRecentFiles(files);
}

std::vector<recentFile> recentFiles;

std::vector<Session*> sessions;

static GLuint backgroundTexture = 0;

static bool firstEverLaunch = false;

static void calcFPS()
{
    static double lastTime = glfwGetTime();
    static uint nbFrames = 0;
    double currentTime = glfwGetTime();
    double delta = currentTime - lastTime;

    nbFrames++;
    
    if (delta >= 1.0) {
        fps = (double)nbFrames / delta;

        nbFrames = 0;
        lastTime = currentTime;
    }
}

static void glfw_error_callback(int error, const char *description)
{
    DispError("GLFW", "%s%d%s%s", "Error ", error, ": ", description);
}

void loadTheme() {
    bool createConfig = false;
    if (std::filesystem::exists("theme.cfg")) {
        try {
            gConfig.Load("theme.cfg");
        } catch (...) {
            DispError("Birch", "Failed to load theme.cfg");
            createConfig = true;
        }
    } else {
        createConfig = true;
    }

    if (createConfig) {
        ImGui::StyleColorsDark();
        ImPlot::StyleColorsDark();

        const auto style = ImGui::GetStyle();

        // Style
        theme.AddItem("Sty.Alpha",                      style.Alpha);
        theme.AddItem("Sty.DisabledAlpha",              style.DisabledAlpha);
        theme.AddItem("Sty.WindowPadding",              style.WindowPadding);
        theme.AddItem("Sty.WindowRounding",             5.0f);
        theme.AddItem("Sty.WindowBorderSize",           style.WindowBorderSize);
        theme.AddItem("Sty.WindowMinSize",              style.WindowMinSize);
        theme.AddItem("Sty.WindowTitleAlign",           style.WindowTitleAlign);
        theme.AddItem("Sty.ChildRounding",              5.0f);
        theme.AddItem("Sty.ChildBorderSize",            style.ChildBorderSize);
        theme.AddItem("Sty.PopupRounding",              5.0f);
        theme.AddItem("Sty.PopupBorderSize",            style.PopupBorderSize);
        theme.AddItem("Sty.FramePadding",               framePad);
        theme.AddItem("Sty.FrameRounding",              5.0f);
        theme.AddItem("Sty.FrameBorderSize",            style.FrameBorderSize);
        theme.AddItem("Sty.ItemSpacing",                style.ItemSpacing);
        theme.AddItem("Sty.ItemInnerSpacing",           itemSpace);
        theme.AddItem("Sty.CellPadding",                style.CellPadding);
        theme.AddItem("Sty.TouchExtraPadding",          style.TouchExtraPadding);
        theme.AddItem("Sty.IndentSpacing",              style.IndentSpacing);
        theme.AddItem("Sty.ColumnsMinSpacing",          style.ColumnsMinSpacing);
        theme.AddItem("Sty.ScrollbarSize",              style.ScrollbarSize);
        theme.AddItem("Sty.ScrollbarRounding",          10.0f);
        theme.AddItem("Sty.GrabMinSize",                style.GrabMinSize);
        theme.AddItem("Sty.GrabRounding",               5.0f);
        theme.AddItem("Sty.LogSliderDeadzone",          style.LogSliderDeadzone);
        theme.AddItem("Sty.TabRounding",                style.TabRounding);
        theme.AddItem("Sty.TabBorderSize",              style.TabBorderSize);
        theme.AddItem("Sty.TabMinWidthForCloseButton",  style.TabMinWidthForCloseButton);
        theme.AddItem("Sty.ButtonTextAlign",            style.ButtonTextAlign);
        theme.AddItem("Sty.SelectableTextAlign",        style.SelectableTextAlign);
        theme.AddItem("Sty.SeparatorTextBorderSize",    style.SeparatorTextBorderSize);
        theme.AddItem("Sty.SeparatorTextAlign",         style.SeparatorTextAlign);
        theme.AddItem("Sty.SeparatorTextPadding",       style.SeparatorTextPadding);
        theme.AddItem("Sty.DisplayWindowPadding",       style.DisplayWindowPadding);
        theme.AddItem("Sty.DisplaySafeAreaPadding",     style.DisplaySafeAreaPadding);
        theme.AddItem("Sty.DockingSeparatorSize",       style.DockingSeparatorSize);
        theme.AddItem("Sty.MouseCursorScale",           style.MouseCursorScale);
        theme.AddItem("Sty.AntiAliasedLines",           style.AntiAliasedLines);
        theme.AddItem("Sty.AntiAliasedLinesUseTex",     style.AntiAliasedLinesUseTex);
        theme.AddItem("Sty.AntiAliasedFill",            style.AntiAliasedFill);
        theme.AddItem("Sty.CurveTessellationTol",       style.CurveTessellationTol);
        theme.AddItem("Sty.CircleTessellationMaxError", style.CircleTessellationMaxError);

        // Colors
        theme.AddItem("Col.Text",                       style.Colors[ImGuiCol_Text]);
        theme.AddItem("Col.TextDisabled",               style.Colors[ImGuiCol_TextDisabled]);
        theme.AddItem("Col.WindowBg",                   style.Colors[ImGuiCol_WindowBg]);
        theme.AddItem("Col.ChildBg",                    style.Colors[ImGuiCol_ChildBg]);
        theme.AddItem("Col.PopupBg",                    style.Colors[ImGuiCol_PopupBg]);
        theme.AddItem("Col.Border",                     style.Colors[ImGuiCol_Border]);
        theme.AddItem("Col.BorderShadow",               style.Colors[ImGuiCol_BorderShadow]);
        theme.AddItem("Col.FrameBg",                    style.Colors[ImGuiCol_FrameBg]);
        theme.AddItem("Col.FrameBgHovered",             style.Colors[ImGuiCol_FrameBgHovered]);
        theme.AddItem("Col.FrameBgActive",              style.Colors[ImGuiCol_FrameBgActive]);
        theme.AddItem("Col.TitleBg",                    style.Colors[ImGuiCol_TitleBg]);
        theme.AddItem("Col.TitleBgActive",              style.Colors[ImGuiCol_TitleBgActive]);
        theme.AddItem("Col.TitleBgCollapsed",           style.Colors[ImGuiCol_TitleBgCollapsed]);
        theme.AddItem("Col.MenuBarBg",                  style.Colors[ImGuiCol_MenuBarBg]);
        theme.AddItem("Col.ScrollbarBg",                style.Colors[ImGuiCol_ScrollbarBg]);
        theme.AddItem("Col.ScrollbarGrab",              style.Colors[ImGuiCol_ScrollbarGrab]);
        theme.AddItem("Col.ScrollbarGrabHovered",       style.Colors[ImGuiCol_ScrollbarGrabHovered]);
        theme.AddItem("Col.ScrollbarGrabActive",        style.Colors[ImGuiCol_ScrollbarGrabActive]);
        theme.AddItem("Col.CheckMark",                  style.Colors[ImGuiCol_CheckMark]);
        theme.AddItem("Col.SliderGrab",                 style.Colors[ImGuiCol_SliderGrab]);
        theme.AddItem("Col.SliderGrabActive",           style.Colors[ImGuiCol_SliderGrabActive]);
        theme.AddItem("Col.Button",                     style.Colors[ImGuiCol_Button]);
        theme.AddItem("Col.ButtonHovered",              style.Colors[ImGuiCol_ButtonHovered]);
        theme.AddItem("Col.ButtonActive",               style.Colors[ImGuiCol_ButtonActive]);
        theme.AddItem("Col.Header",                     style.Colors[ImGuiCol_Header]);
        theme.AddItem("Col.HeaderHovered",              style.Colors[ImGuiCol_HeaderHovered]);
        theme.AddItem("Col.HeaderActive",               style.Colors[ImGuiCol_HeaderActive]);
        theme.AddItem("Col.Separator",                  style.Colors[ImGuiCol_Separator]);
        theme.AddItem("Col.SeparatorHovered",           style.Colors[ImGuiCol_SeparatorHovered]);
        theme.AddItem("Col.SeparatorActive",            style.Colors[ImGuiCol_SeparatorActive]);
        theme.AddItem("Col.ResizeGrip",                 style.Colors[ImGuiCol_ResizeGrip]);
        theme.AddItem("Col.ResizeGripHovered",          style.Colors[ImGuiCol_ResizeGripHovered]);
        theme.AddItem("Col.ResizeGripActive",           style.Colors[ImGuiCol_ResizeGripActive]);
        theme.AddItem("Col.Tab",                        style.Colors[ImGuiCol_Tab]);
        theme.AddItem("Col.TabHovered",                 style.Colors[ImGuiCol_TabHovered]);
        theme.AddItem("Col.TabActive",                  style.Colors[ImGuiCol_TabActive]);
        theme.AddItem("Col.TabUnfocused",               style.Colors[ImGuiCol_TabUnfocused]);
        theme.AddItem("Col.TabUnfocusedActive",         style.Colors[ImGuiCol_TabUnfocusedActive]);
        theme.AddItem("Col.DockingPreview",             style.Colors[ImGuiCol_DockingPreview]);
        theme.AddItem("Col.DockingEmptyBg",             style.Colors[ImGuiCol_DockingEmptyBg]);
        theme.AddItem("Col.PlotLines",                  style.Colors[ImGuiCol_PlotLines]);
        theme.AddItem("Col.PlotLinesHovered",           style.Colors[ImGuiCol_PlotLinesHovered]);
        theme.AddItem("Col.PlotHistogram",              style.Colors[ImGuiCol_PlotHistogram]);
        theme.AddItem("Col.PlotHistogramHovered",       style.Colors[ImGuiCol_PlotHistogramHovered]);
        theme.AddItem("Col.TableHeaderBg",              style.Colors[ImGuiCol_TableHeaderBg]);
        theme.AddItem("Col.TableBorderStrong",          style.Colors[ImGuiCol_TableBorderStrong]);
        theme.AddItem("Col.TableBorderLight",           style.Colors[ImGuiCol_TableBorderLight]);
        theme.AddItem("Col.TableRowBg",                 style.Colors[ImGuiCol_TableRowBg]);
        theme.AddItem("Col.TableRowBgAlt",              style.Colors[ImGuiCol_TableRowBgAlt]);
        theme.AddItem("Col.TextSelectedBg",             style.Colors[ImGuiCol_TextSelectedBg]);
        theme.AddItem("Col.DragDropTarget",             style.Colors[ImGuiCol_DragDropTarget]);
        theme.AddItem("Col.NavHighlight",               style.Colors[ImGuiCol_NavHighlight]);
        theme.AddItem("Col.NavWindowingHighlight",      style.Colors[ImGuiCol_NavWindowingHighlight]);
        theme.AddItem("Col.NavWindowingDimBg",          style.Colors[ImGuiCol_NavWindowingDimBg]);
        theme.AddItem("Col.ModalWindowDimBg,",          style.Colors[ImGuiCol_ModalWindowDimBg]);

        theme.Save("theme.cfg");
    }

}

void loadConfig() {
    bool createConfig = false;
    if (std::filesystem::exists("config.cfg")) {
        try {
            gConfig.Load("config.cfg");
        } catch (...) {
            DispError("Birch", "Failed to load config.cfg");
            createConfig = true;
        }
    } else {
        createConfig = true;
    }

    if (createConfig) {
        gConfig.AddItem("PreviewSize", 5000);
        gConfig.AddItem("ShowMinimap", true);
        gConfig.AddItem("PlotErrors",  true);
        gConfig.AddItem("ErrorAlpha",  0.25f);
        gConfig.AddItem("LiveLoad",    false);
        gConfig.AddItem("PlotsShown",  5);
        gConfig.AddItem("LoadTheme",   true);
        gConfig.AddItem("Style",       "theme.cfg");
        gConfig.AddItem("Icons",       "theme.bpk");

        gConfig.Save("config.cfg");
    }

    signalMaxSize = gConfig["PreviewSize"].Int();
}

// Sets up ImGui styling
void setStyling()
{
    ImGui::GetStyle().WindowRounding = 5.0f;
    ImGui::GetStyle().FrameRounding = 5.0f;
    ImGui::GetStyle().FramePadding = framePad;
    ImGui::GetStyle().ItemInnerSpacing = itemSpace;

    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();

    ImGui::GetStyle().PopupRounding = 5.0f;
    ImGui::GetStyle().ScrollbarRounding = 10.0f;
    ImGui::GetStyle().GrabRounding = 5.0f;
    ImGui::GetStyle().ChildRounding = 5.0f;

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImGui::GetStyleColorVec4(ImGuiCol_Button));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 0.8));

    ImPlot::PushStyleColor(ImPlotCol_XAxis, ImVec4(1, 1, 1, 0.8));
    ImPlot::PushStyleColor(ImPlotCol_YAxis, ImVec4(1, 1, 1, 0.8));
    ImPlot::PushStyleColor(ImPlotCol_YAxis2, ImVec4(1, 0.8, 1, 0.8));
    ImPlot::PushStyleColor(ImPlotCol_YAxis3, ImVec4(1, 1, 0.8, 0.8));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.066, 0.066, 0.066, 1));

    ImPlot::PushStyleColor(ImPlotCol_Selection, ImVec4(0.2, 0.8, 1, 1));

    ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0, 0));

    ImVec4 mb = ImGui::GetStyle().Colors[ImGuiCol_MenuBarBg];
    menuBack = ImColor(mb.x, mb.y, mb.z, 1.0f);
}
static void buildFonts() {
    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    builder.AddRanges(imGuiIO->Fonts->GetGlyphRangesDefault());
    builder.AddText(u8"ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω₂⌈∛⌉");
    builder.BuildRanges(&ranges);

    imGuiIO->Fonts->AddFontFromMemoryCompressedTTF(font_compressed_data,      font_compressed_size,      16.0f, nullptr, ranges.Data);
    imGuiIO->Fonts->AddFontFromMemoryCompressedTTF(font_bold_compressed_data, font_bold_compressed_size, 16.0f, nullptr, ranges.Data);

    // Load all fonts from fonts folder
    std::filesystem::path fontsPath = std::filesystem::current_path() / "fonts";

    try {
        for (const auto& entry : std::filesystem::directory_iterator(fontsPath)) {
            if (entry.path().extension() == ".ttf") {
                imGuiIO->Fonts->AddFontFromFileTTF(entry.path().string().c_str(), 16.0f, nullptr, ranges.Data);
            }
        }
    } catch (...) {
        DispError("buildFonts", "fonts directory could not be found!");
    }

    imGuiIO->Fonts->Build();

    font_normal = imGuiIO->Fonts->Fonts[0];
    font_bold   = imGuiIO->Fonts->Fonts[1];

    imGuiIO->FontGlobalScale = 1.0f;
}

static GLuint createNoise(uint w, uint h) {
    uint* pixels = new uint[w * h];

    #pragma omp parallel for
    for (uint i = 0; i < w * h; i++) {
        // Random greyscale between 20 and 28
        uchar r = (uchar)(rand() % 8 + 18);
        pixels[i] = (uchar)255 << 24 | r << 16 | r << 8 | r;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R,     GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_REPEAT);

    delete[] pixels;

    return tex;
}

//loads glfw, ImGui, UI Textures, Tools, Widgets, and gradients
//todo: this needs to be split up
bool initWindow()
{
#ifdef FFT_IMPL_CUDA
    cuda_init();
#endif

    // Build window functions
    GetWindows();

    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
    glfwWindowHint(GLFW_SAMPLES, 2);

    // Create window with graphics context
    window = glfwCreateWindow(1920, 1080, "Birch", NULL, NULL);

    if (window == NULL)
        return false;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (gl3wInit()) {
        DispError("GL3W", "%s", "Failed to initialize OpenGL\n");
        return false;
    }


    IMGUI_CHECKVERSION();
    context = ImGui::CreateContext();
    imGuiIO = &ImGui::GetIO();

    imGuiIO->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    //imGuiIO->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    //imGuiIO->ConfigViewportsNoDecoration = false;
    //imGuiIO->ConfigViewportsNoAutoMerge = true;

    buildFonts();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    ImPlot::CreateContext();

    loadConfig();
    loadTheme();
    setStyling();

    for (auto& p : plugins) {
        if (p.Type == PluginType::IQProcessor) {
            auto proc = (Birch::PluginIQProcessor*)p.Instance();
            proc->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };
            proc->Math = &bmath;

            proc->Init();

            if (proc->Tool()) {
                pluginTools.push_back(new IQProcHost(p.Name, "", proc));
            }
        }
    }
#ifndef NO_PYTHON
    for (auto& p : pythonIQPlugins) {
        auto proc = new PythonIQProcessorHost(p);
        proc->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };

        proc->Init();

        if (proc->Tool()) {
            pluginTools.push_back(new IQProcHost(proc->Name(), "", proc));
        }
    }

    for (auto& p : pythonTBDPlugins) {
        auto proc = new PythonTBDProcessorHost(p);
        proc->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };

        proc->Init();

        if (proc->Tool()) {
            pluginTools.push_back(new TBDProcHost(proc->Name(), "", proc));
        }
    }
#endif

    // Load icons
    PackFile* pk;
    try {
        pk = new PackFile("theme.bpk");

        const auto& keys = pk->Keys();
        for (const auto& key : keys)
            textures[key] = LoadSVG((*pk)[key]->toString().c_str(), 32 * iconScale, 32 * iconScale, iconScale);
    } catch (...) {
        DispError("Birch", "Failed to open theme.bpk");
        return false;
    }
    delete pk;

    // Load plot markers
    PackFile* pk2;
    try {
        pk2 = new PackFile("markers.bpk");

        const auto& keys = pk2->Keys();
        for (const auto& key : keys)
            markers[key] = (*pk2)[key]->toString();
    } catch (...) {
        DispError("Birch", "Failed to open markers.bpk");
        return false;
    }
    delete pk2;

    DispInfo("Birch", "Loaded %d markers", (int)markers.size());

    // Load gradients
    PackFile* pk3;
    try {
        pk3 = new PackFile("gradients.bpk");

        const auto& keys = pk3->Keys();
        for (const auto& key : keys) {
            auto* g = Gradient::Load((*pk3)[key]->toString().c_str());
            if (g != nullptr)
                gradients.push_back(g);
        }
    } catch (...) {
        DispError("Birch", "Failed to open gradients.bpk");
        return false;
    }
    delete pk3;
    
    // Sort gradients by catagory
    std::sort(gradients.begin(), gradients.end(), [](Gradient* a, Gradient* b) {
        return a->CatagoryID < b->CatagoryID;
    });

    DispInfo("Birch", "Loaded %d gradients", (int)gradients.size());

    // Load shader code
    Stopwatch shaderTime;

    PackFile* pk4;
    try {
        pk4 = new PackFile("shaders.bpk");
    } catch (...) {
        DispError("Birch", "Failed to open shaders.bpk");
        return false;
    }

    // Compile shaders
    try {
        Shader* pltShader = new Shader((*pk4)["shaders/spectra.vert"]->toString(), (*pk4)["shaders/spectra.frag"]->toString());
        gShaders["spectra"] = pltShader;
    } catch (...) {
        DispError("Birch", "Failed to compile spectra shader");
        return false;
    }

    try {
        Shader* pltShader = new Shader((*pk4)["shaders/scatter.vert"]->toString(), (*pk4)["shaders/scatter.frag"]->toString());
        gShaders["scatter"] = pltShader;
    } catch (...) {
        DispError("Birch", "Failed to compile scatterplot shader");
        return false;
    }

    try {
        Shader* pltShader = new Shader((*pk4)["shaders/liveSpectra.vert"]->toString(), (*pk4)["shaders/liveSpectra.frag"]->toString());
        gShaders["liveSpectra"] = pltShader;
    } catch (...) {
        DispError("Birch", "Failed to compile liveSpectra shader");
        return false;
    }

    // I suppose there may be more later....

    delete pk4;

    DispInfo("Birch", "Compiled shaders in %.3lfs", shaderTime.Now());

    InitPlotterView(&gradients, &pluginTools, &nextID, imGuiIO, &progressBarPercent, iconScale, 0, &textures);
    InitHistView(&gradients, &pluginTools, &nextID, imGuiIO, &progressBarPercent, iconScale);

    // Builds marker previews
    InitPlotFieldRenderers();

    //views.push_back(new MissionControl());
    //views.push_back(new Plotter());

    sessions.push_back(new Session(&progressBarPercent));

    // Generate background texture
    backgroundTexture = createNoise(512, 512);

    return true;
}

static void loadProject(const char* path, int selectedView) {
    DispInfo("Birch", "Loading project %s", path);

    Project p = Project(path);

    for (auto& file : p.Files()) {
    bool good = false;
    for (auto& plugin : plugins) {
        if (!strncmp(plugin.Name, file.plugin.c_str(), 128)) {
            std::string sep = "/";
            #ifdef _WIN64
                sep = "\\";
            #endif

            auto fp = p.Directory() + sep + file.path;

            DispInfo("Birch", "Opening file %s with %s", fp.c_str(), plugin.Name);

            auto inst = plugin.Instance();
            inst->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };

            char* ugh = strdup(fp.c_str());
            //((Plotter*)views[selectedView])->AddFile(ugh, inst, file.openFields);
            free(ugh);

            good = true;
            break;
        }
    }
    if (!good)
        DispError("Birch", "Could not find plugin %s", file.plugin.c_str());
    }
}

static void loadFile(PluginInterface& plugin, Session* session, const std::string& path = "") {
    auto* tempInst = plugin.Instance();

    bool needsPath = true;

    const char  defaultList[] = "";
    const char* filterList    = nullptr;

    if (tempInst->Type == PluginType::IQGetter) {
        Birch::PluginIQGetter* inst = (Birch::PluginIQGetter*)tempInst;

        filterList = inst->FileExtensions();
        needsPath  = !inst->NoFilePath();
    }
    else {
        Birch::PluginTBDGetter* inst = (Birch::PluginTBDGetter*)tempInst;

        filterList = inst->FileExtensions();
        needsPath  = !inst->NoFilePath();
    }

    if (filterList == nullptr)
        filterList = defaultList;

    delete tempInst;


    const bool addView = session->Files()->size() == 0 && session->Views()->size() == 0;


    if (!needsPath) {
        auto inst = plugin.Instance();
        BFile* file = nullptr;

        DispInfo("Birch", "Opening %s", plugin.Name);
        inst->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };

        session->AddFile(nullptr, &file, inst);

        session->ReloadFields();

        if (inst->Type == PluginType::TBDGetter) {
            auto tbdInst = (Birch::PluginTBDGetter*)tempInst;

            tbdInst->_bff = file;
            tbdInst->_bfs = session;

            tbdInst->Update = [](const Birch::PluginTBDGetter* plug) -> void {
                if (plug->_bff == nullptr)
                    return;
                if (plug->_bfs == nullptr)
                    return;

                Session* session = (Session*)plug->_bfs;

                session->ReloadFields();
            };
        }

        // TODO: Probably should be a setting later
        if (addView) {
            session->AddView(new Plotter(session));
        }

        return;
    }


    nfdpathset_t files;
    nfdresult_t result;
    
    if (path == "") {
        result = NFD_OpenDialogMultiple(filterList, NULL, &files);
    }
    else {
        files.count = 1;
        files.buf = (nfdchar_t*)malloc(path.size() + 1);
        strcpy(files.buf, path.c_str());
        files.indices = (size_t*)malloc(sizeof(size_t));
        files.indices[0] = 0;
    }

    // Open a file
    if (result == NFD_OKAY || path != "")
    {
        for (int i = 0; i < files.count; i++)
        {
            auto inst = plugin.Instance();

            DispInfo("Birch", "Opening file with %s", plugin.Name);
            inst->SubmitSVG = [](const char* svg, unsigned w, unsigned h) { return LoadSVG(svg, w, h); };

            if (inst->Type == PluginType::TBDGetter) {
                auto tbdInst = (Birch::PluginTBDGetter*)tempInst;

                tbdInst->Update = [](const Birch::PluginTBDGetter* plug) -> void {
                    BFile* file = (BFile*)plug->_bff;
                };
            }

            auto fields = std::vector<std::string>();
            //plotter->AddFile(NFD_PathSet_GetPath(&files, i), inst, fields);
            BFile* file = nullptr;
            session->AddFile(NFD_PathSet_GetPath(&files, i), &file, inst);

            if (file != nullptr) {
                addRecentFile(recentFiles, file->Filepath(), file->Filename(), plugin.Name);

                // TODO: Probably should be a setting later
                if (addView) {
                    session->AddView(new Plotter(session));
                }
            }
        }
    }
    else
    {
        DispWarning("main", "%s", "No file was selected");
    }

    NFD_PathSet_Free(&files);
}

int display_w, display_h;

int main_(int argc, char **args)
{
    bool running = true;
    bool startup = true;

    //int selectedView = 0;

    while (running)
    {
        running = !glfwWindowShouldClose(window);

#ifndef SERVER
        auto prog = *gProgressBar;
        if (prog > 0 && prog < 0.99f)
            glfwPollEvents();
        else
            glfwWaitEventsTimeout(1.0 / 24);
#else
        glfwWaitEvents();
#endif

        if (gMainThreadTasks.size() > 0)
        {
            for (auto& task : gMainThreadTasks) {
                task->Task(task->args);
                delete task;
            }

            gMainThreadTasks.clear();
        }

        int width, height;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::GetStyle().ItemSpacing = itemSpace;

        //const ImGuiViewport* viewport = ImGui::GetMainViewport();
        //ImGui::SetNextWindowPos(viewport->WorkPos);
        //ImGui::SetNextWindowSize(viewport->WorkSize);
        //ImGui::SetNextWindowViewport(viewport->ID);

        //printf("%x\n", viewport->ID);

        if (startup) {
            if (argc > 1) {
                if (strncmp(args[1], "-p", 2) == 0) {
                    signalMaxSize = atoi(args[2]);
                } else {
                    loadProject(args[1], 0);
                }
            }
        }

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
        //ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        //ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(display_w, display_h), ImGuiCond_Always);

        if (ImGui::Begin("Birch", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking))
        {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(2);

            const auto window = ImGui::GetCurrentWindow();

            ImGui::GetWindowDrawList()->AddImage(t_ImTexID(backgroundTexture), window->Pos, window->Size, ImVec2(0, 0), ImVec2(512, 512));

            /*** Menu for when no files are open

            if (sessions.size() == 0 || sessions[0]->Files()->size() == 0) {
                uint offset = 64;
                ImGui::SetCursorPos(ImVec2(offset - 24, offset));

                // Set style to gray
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1, 0.1, 0.1, 0.5));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.5, 0.5, 0.5, 1.0));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
                if (ImGui::BeginChildFrame(ImGui::GetID("MainMenu_"), ImVec2(400, display_h - offset * 2))) {
                    ImGui::PopStyleColor(2);
                    ImGui::PopStyleVar();

                    ImGui::Button("Open Project");
                    ImGui::SameLine();
                    ImGui::Button("Import");

                    ImGui::Separator();

                    ImGui::Text("Recent Files");

                    if (ImGui::BeginListBox("##RecentFiles", ImVec2(-1, -1))) {
                        if (recentFiles.size() == 0)
                            ImGui::Selectable("(Empty)", false, ImGuiSelectableFlags_Disabled);
                        
                        for (const auto& f : recentFiles) {
                            ImGui::Selectable(f.filename.c_str());
                        }

                        ImGui::EndListBox();
                    }
                    
                } else {
                    ImGui::PopStyleColor(2);
                    ImGui::PopStyleVar();
                }
                ImGui::EndChildFrame();
            }

            */

            bool openAbout = false;
            static bool openDebug[] = { false, false, false, false };
            // std::vector<bool> is stupid
            static std::vector<char> shaderEdit = std::vector<char>(gShaders.size(), false);

            bool openLicense = false || firstEverLaunch;
            static bool showStyleEditor = false;

            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

            if (ImGui::BeginMainMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open Project")) {
                        nfdchar_t *outPath = NULL;
                        nfdresult_t result = NFD_OpenDialog("bpf", NULL, &outPath);
                        
                        if (result == NFD_OKAY) {
                            loadProject(outPath, 0);
                            free(outPath);
                        }
                    }

                    if (ImGui::BeginMenu("Import")) {
                        if (ImGui::BeginMenu("IQ")) {
                            for (auto& p : plugins)
                                if (p.Type == PluginType::IQGetter)
                                    if (ImGui::MenuItem(p.Name))
                                        loadFile(p, sessions[0]);

                            ImGui::EndMenu();
                        }

                        if (ImGui::BeginMenu("TBD")) {
                            for (auto& p : plugins)
                                if (p.Type == PluginType::TBDGetter)
                                    if (ImGui::MenuItem(p.Name))
                                        loadFile(p, sessions[0]);

                            ImGui::EndMenu();
                        }

                        ImGui::EndMenu();
                    }

                    ImGui::Separator();

                    if (ImGui::BeginMenu("Recent")) {
                        if (recentFiles.size() == 0)
                            ImGui::Selectable("(Empty)", false, ImGuiSelectableFlags_Disabled);

                        for (const auto& f : recentFiles) {
                            if (ImGui::MenuItem(f.filename.c_str())) {
                                // Find plugin
                                bool found = false;
                                for (auto& p : plugins) {
                                    if (std::string(p.Name) == f.pluginName) {
                                        loadFile(p, sessions[0], f.path);
                                        found = true;
                                        break;
                                    }
                                }

                                if (!found) {
                                    DispError("Birch", "Could not find plugin %s", f.pluginName.c_str());
                                }
                            }
                            if (ImGui::BeginItemTooltip()) {
                                ImGui::Text("%s", f.path.c_str());
                                ImGui::TextDisabled("%s", f.pluginName.c_str());
                                ImGui::EndTooltip();
                            }
                        }

                        ImGui::Separator();

                        if (ImGui::MenuItem("Clear")) {
                            recentFiles.clear();
                            saveRecentFiles(recentFiles);
                        }

                        ImGui::EndMenu();
                    }

                    ImGui::Separator();

                    if (ImGui::BeginMenu("Settings"))
                    {
                        // This aught to just launch a settings window

                        if (ImGui::BeginMenu("Theme"))
                        {
                            if (ImGui::MenuItem("Light"))
                            {
                                ImGui::StyleColorsLight();
                                ImPlot::StyleColorsLight();
                            }

                            if (ImGui::MenuItem("Dark"))
                            {
                                ImGui::StyleColorsDark();
                                ImPlot::StyleColorsDark();
                            }

                            if (ImGui::MenuItem("Editor"))
                            {
                                showStyleEditor = true;
                            }

                            ImGui::EndMenu();
                        }

                        if (ImGui::MenuItem("Hotkeys"))
                        {
                        }

                        if (ImGui::MenuItem("Plots"))
                        {
                        }

                        if (ImGui::MenuItem("Other"))
                        {
                        }

                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Views"))
                {
                    if (ImGui::MenuItem("Plotter"))
                        sessions[0]->AddView(new Plotter(sessions[0]));

                    if (ImGui::MenuItem("Histogram"))
                        sessions[0]->AddView(new Histogram(sessions[0]));

                    //if (ImGui::MenuItem("Recorder"))
                    //    sessions[0]->AddView(new Live(sessions[0]));

                    

/*
                    if (ImGui::MenuItem("Raster"))
                        views.push_back(new Raster());

                    if (ImGui::MenuItem("Polar"))
                        views.push_back(new Polar());
*/

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Plugins"))
                {
                    if (ImGui::MenuItem("Plugin Manager")) {

                    }

                    ImGui::Separator();

                    if (ImGui::BeginMenu("File Import")) {
                        for (auto& p : plugins) {
                            if (!(p.Type == PluginType::IQGetter || p.Type == PluginType::TBDGetter))
                                continue;

                            if (ImGui::MenuItem(p.Name)) {

                            }
                        }

                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("Tools")) {
                        for (auto& p : plugins) {
                            if (!(p.Type == PluginType::IQProcessor || p.Type == PluginType::TBDProcessor))
                                continue;

                            if (ImGui::MenuItem(p.Name)) {

                            }
                        }

                        ImGui::EndMenu();
                    }
                    
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Debug")) {
                    if (ImGui::MenuItem("Debugger"))
                        openDebug[1] = true;
                    
                    if (ImGui::MenuItem("Plot Debugger"))
                        openDebug[3] = true;

                    if (ImGui::MenuItem("Log"))
                        openDebug[0] = true;
                    
                    if (ImGui::MenuItem("Stack"))
                        openDebug[2] = true;
                    
                    if (ImGui::BeginMenu("Shaders")) {
                        int i = 0;
                        for (auto& s : gShaders) {
                            if (ImGui::MenuItem(s.first.c_str())) {
                                shaderEdit[i] = true;
                            }
                            i++;
                        }

                        ImGui::EndMenu();
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("About"))
                {
                    if (ImGui::MenuItem("About Birch"))
                        openAbout = true;

                    if (ImGui::MenuItem("License"))
                        openLicense = true;

                    ImGui::EndMenu();
                }

                ImGui::EndMainMenuBar();
            }

            ImGui::PopStyleVar();

            if (openAbout)
            {
                ImGui::OpenPopup("About");
            }
            if (openLicense)
            {
                firstEverLaunch = false;
                ImGui::OpenPopup("License");
            }
            if (showStyleEditor)
            {
                if (ImGui::Begin("Style Editor", &showStyleEditor)) {
                    ImGui::ShowStyleEditor();
                }
                ImGui::End();
                if (ImGui::Begin("Plot Style Editor", &showStyleEditor)) {
                    ImPlot::ShowStyleEditor();
                }
                ImGui::End();
            }

            if (openDebug[1])
                ImGui::ShowMetricsWindow(&openDebug[1]);
            if (openDebug[0])
                ImGui::ShowDebugLogWindow(&openDebug[0]);
            if (openDebug[2])
                ImGui::ShowStackToolWindow(&openDebug[2]);
            if (openDebug[3])
                ImPlot::ShowMetricsWindow(&openDebug[3]);

            int idx = 0;
            for (auto& s : gShaders) {
                if (shaderEdit[idx]) {
                    bool open = shaderEdit[idx];
                    s.second->ShowEditor(&open);
                    shaderEdit[idx] = open;
                }
                idx++;
            }

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);

            ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
            ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
            bool dummy = true;
            if (ImGui::BeginPopupModal("About", &dummy, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

                ImGui::Text("Birch Signals Analysis Tool");
                ImGui::Text("© 2020-2023 Levi Miller");

                ImGui::NewLine();

                ImGui::Text("Version %s", BSAT_VERSION);
                ImGui::Text("-");
                ImGui::Text("ImGui %s", IMGUI_VERSION);
                ImGui::Text("ImPlot %s", IMPLOT_VERSION);

                ImGui::NewLine();

                ImGui::Text("System Configuration:");
#ifdef _WIN64
                ImGui::Text("Platform: WIN64");
#endif
#ifdef __APPLE__
                ImGui::Text("Platform: macOS");
#endif

                ImGui::Text("Renderer: OpenGL 4.6");

                ImGui::NewLine();

                ImGui::Text("Birch uses the following libraries:");
                ImGui::Text("GLFW");
                ImGui::Text("ImGui");
                ImGui::Text("ImPlot");
                ImGui::Text("Freetype");
                ImGui::Text("nanoSVG");
                ImGui::Text("libNFD");
                ImGui::PopStyleVar();

                ImGui::EndPopup();
            }

            ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_Appearing);
            if (ImGui::BeginPopupModal("License", NULL))
            {
                if (ImGui::BeginTabBar("licenseTabs"))
                {
                    if (ImGui::BeginTabItem("Birch"))
                    {
                        ImGui::Text("Birch Signals Analysis Tool © Levi Miller, 2023");
                        ImGui::NewLine();
                        ImGui::Text("By using this software, you agree to the terms below:");

                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseBirch);
                        ImGui::EndChildFrame();

                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("ImGui"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseImGui);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("ImPlot"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseImPlot);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("GLFW"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseGLFW);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("GLEW"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseGLEW);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Freetype"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseFreetype);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("NFD"))
                    {
                        if (ImGui::BeginChildFrame(3234, ImVec2(-1, -35)))
                            ImGui::TextWrapped("%s", licenseNFD);
                        ImGui::EndChildFrame();
                        
                        ImGui::EndTabItem();
                    }
                
                    ImGui::EndTabBar();
                }

                if (ImGui::Button("Agree"))
                    ImGui::CloseCurrentPopup();

                ImGui::EndPopup();
            }

            ImGui::PopStyleVar();

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 26);

            ImGuiWindowClass windowClass;

            windowClass.ClassId = ImGui::GetID("ViewDock");
            windowClass.DockingAllowUnclassed = false;
            windowClass.DockingAlwaysTabBar   = false;

            ImGuiID dockspace_id = ImGui::GetID("MainWindowDockspace");
            ImGui::DockSpace(dockspace_id, ImVec2(0, -24), ImGuiDockNodeFlags_PassthruCentralNode, &windowClass);

            auto views = sessions[0]->Views();
            int i = 0;
            for (auto v : *views) {
                bool open = true;

                std::string name = std::string(v->Name()) + "##" + std::to_string(i++);

                ImGui::PushID(v);
                ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_Appearing);
                ImGui::SetNextWindowClass(&windowClass);
                if (ImGui::Begin(name.c_str(), &open)) {
                    v->Render();
                }
                ImGui::End();
                ImGui::PopID();

                if (!open) {
                    views->erase(std::remove(views->begin(), views->end(), v), views->end());
                    delete v;
                    break;
                }
            }

            // Status bar
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(-4, ImGui::GetWindowPos().y + ImGui::GetWindowHeight() - 25), ImVec2(ImGui::GetWindowWidth() + 16, ImGui::GetWindowPos().y + ImGui::GetWindowHeight()), menuBack);

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);

            ImGui::Text(" %u fps | %s | %d plugins | %s", fps, computeString.c_str(), (int)plugins.size(), GetDefaultLogger()->GetMessages()->back().c_str());

            if (ImGui::BeginPopup("Log")) {
                if (ImGui::BeginChildFrame(23423432, ImVec2(512, 256))) {
                    for (auto msg : *GetDefaultLogger()->GetMessages()) {
                        ImGui::Text("%s", msg.c_str());
                    }
                }
                ImGui::EndChildFrame();
                ImGui::EndPopup();
            }

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
                ImGui::OpenPopup("Log");

            if (progressBarPercent < 1 && progressBarPercent > 0) {
                float width = ImGui::GetWindowSize().x;
                auto pos = ImVec2(width - width * .15, ImGui::GetWindowSize().y - 18);

                ImGui::SetCursorPos(pos);
                ImGui::ProgressBar(progressBarPercent, ImVec2(-1, ImGui::GetFrameHeight() - 12), "");
            }

            ImGui::End();
        } else {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(2);
        }

        // Render
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (imGuiIO->ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);

        calcFPS();

        startup = false;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

unsigned long long getTotalSystemMemory()
{
#ifdef __unix__ 
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
#endif
#ifdef __APPLE__ 
    return 0;
#endif
#ifdef _WIN64
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
#else
    return 0;
#endif
}

void shutdown(bool crash = false) {
    if (crash)
        DispMessage("Birch", "Sorry");
    else
        DispMessage("Birch", "Goodbye");
    
    delete GetDefaultLogger();

    SetDefaultLogger(nullptr);
}

void sigsegv_handler(int signum, siginfo_t *info, void *data)
{
    if (GetDefaultLogger() != nullptr) {
        DispError("Birch", "That's no good");
        DispError("Birch", "\t-> code: %d", info->si_code);
        DispError("Birch", "\t-> errno: %d", info->si_errno);
        DispError("Birch", "\t-> signum: %d", info->si_signo);
        DispError("Birch", "\t-> addr: %p", info->si_addr);

        shutdown(true);
    }

    exit(signum);
}

int main(int argc, char **args)
{
    auto initTimer = Stopwatch();

    srand(time(NULL));

    SetDefaultLogger(new Logger(true, Logger::Level::Debug, "birch.log"));

    {
        struct sigaction  handler;
        sigemptyset(&handler.sa_mask);
        handler.sa_sigaction = &sigsegv_handler;
        handler.sa_flags = SA_SIGINFO;

        if (sigaction(SIGSEGV, &handler, NULL) == -1) {
            DispError("Birch", "Cannot set SIGSEGV handler: %s", strerror(errno));
        }
    }

#ifdef _WIN64
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif

    printf("\n%s", ascii_logo);
    DispMessage("Birch", "%s%s", "Starting Birch Signals Analysis Tool v", BSAT_VERSION);
#ifdef _WIN64
    DispInfo("Birch", "%s", "Platform: WIN64 / OpenGL 4.6");
#endif
#ifdef __APPLE__
    DispInfo("Birch", "%s", "Platform: Apple / OpenGL 4.1 Core");
#endif
    DispInfo("Birch", "%s", "Build: " __DATE__ " " __TIME__);

    DispInfo("Birch", "%s%d", "Compute device 0: CPUx", std::thread::hardware_concurrency());

    computeString = "CPUx" + std::to_string(std::thread::hardware_concurrency());

#ifdef FFT_IMPL_CUDA
    size_t totalMem = 0;
    uint cudaDeviceCount = cuda_count_devices();
    if (cudaDeviceCount > 0)
        cuda_list_devices();

    for (uint i = 0; i < cudaDeviceCount; i++) {
        size_t gpuMem = 0;
        size_t freeMem = 0;

        cuda_get_device_memory(i, &freeMem, &gpuMem);

        totalMem += gpuMem;
    }

    computeString += ", GPUx" + std::to_string(cudaDeviceCount);

    DispInfo("Birch", "%s%.2f%s", "GPU memory: ", totalMem / (1024 * 1024 * 1024.0f), "GiB");
#endif

    totalMem = getTotalSystemMemory();

    computeString += ", " + std::to_string((uint)(totalMem / (1024 * 1024 * 1024.0f))) + "GiB";

    DispInfo("Birch", "%s%.2f%s", "System memory: ", totalMem / (1024 * 1024 * 1024.0f), "GiB");
/* TEMP
    #pragma omp parallel
    {
        #pragma omp single
        DispInfo("Birch", "OpenMP Threads: %d", omp_get_num_threads());
    }
*/
    DispInfo("Birch", "FFT Provider: %s", fft_get_impl());

    char* plugDir = getenv("BIRCH_PLUGINS");
    bool freePlugDir = false;

    if (plugDir == nullptr) {
        DispWarning("Birch", "BIRCH_PLUGINS is not defined! Searching for plugins in '%s'", std::filesystem::absolute("plugins").c_str());
        plugDir = strdup("plugins");
        freePlugDir = true;
    }

        std::string pyDir = std::string(plugDir);

#ifdef _WIN64
        pyDir += "\\python";
#else
        pyDir += "/python";
#endif
#ifndef NO_PYTHON
    if (std::filesystem::exists(pyDir)) {
        Stopwatch pyInit;

        InitPython(pyDir.c_str());

        DispInfo("Birch", "Initialized Python in %.3lfs", pyInit.Now());
    }
#endif
    Stopwatch pluginLoad;
    DispInfo("Birch", "Loading plugins...");

    if (std::filesystem::exists(plugDir)) {
        plugins = LoadPlugins(plugDir);
        for (auto& p : plugins) {
            const char* typeName[] = { "IO / IQ", "IO / TBD", "IQ Processor" };
            DispInfo("Birch", "\t-> %s | %s", p.Name, typeName[(int)p.Type]);
        }

#ifndef NO_PYTHON
        if (std::filesystem::exists(pyDir)) {
            pythonIQPlugins = LoadPythonIQProcessors(pyDir.c_str());

            for (auto& p : pythonIQPlugins) {
                DispInfo("Birch", "\t-> %s | %s", p->Name().c_str(), "IQ Processor (Python)");
            }

            if (pythonIQPlugins.size() == 0) {
                DispInfo("Birch", "\t-> No Python IQ plugins found");
            }

            pythonTBDPlugins = LoadPythonTBDProcessors(pyDir.c_str());

            for (auto& p : pythonTBDPlugins) {
                DispInfo("Birch", "\t-> %s | %s", p->Name().c_str(), "TBD Processor (Python)");
            }

            if (pythonTBDPlugins.size() == 0) {
                DispInfo("Birch", "\t-> No Python TBD plugins found");
            }
        }
#endif
    } else {
        DispError("Birch", "Plugin directory was not found! No plugins will be loaded.");
        plugins = std::vector<PluginInterface>();
    }

    if (freePlugDir)
        free(plugDir);

    DispInfo("Birch", "Loaded %d plugins in %.3lfs", (int)plugins.size(), pluginLoad.Now());

    // Check if this is the first launch
    firstEverLaunch = !std::filesystem::exists("imgui.ini");

    if (!initWindow()) {
        DispError("Birch", "Init failed");
        return 1;
    }

    loadRecentFiles(recentFiles);

    DispInfo("Birch", "Initialized in %.3lfs", initTimer.Now());

    main_(argc, args);

#ifndef NO_PYTHON
    ShutdownPython();
#endif

    shutdown();

    return 0;
}

#include "plotRenderer.hpp"

#include "implot.h"
#include <GL/gl3w.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <string>
#include <map>
#include <unordered_map>

using std::string;
using std::map;
using std::vector;

#include "include/birch.h"
#include "logger.hpp"
#include "widgets.hpp"
#include "fboHelpers.hpp"
#include "svg.hpp"
#include "shader.hpp"

#include "defines.h"

#include "implot_internal.h"
#include "ImPlot_Extras.h"

// Load stbi_image.h
#include "stb_image.h"

#ifdef FFT_IMPL_CUDA
#include "plotRenderer.cuh"
#endif

using namespace Birch;

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"

uint uploadTexture(const uint* pixels, uint height, uint width) {
    uint tex;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    return tex;
}


template<typename K, typename V>
vector<K> getMapKeys(const std::map<K, V>& m) {
    std::vector<K> v;
    v.reserve(m.size());

    std::transform(m.begin(), m.end(), std::back_inserter(v), [](const auto& p) { return p.first; });

    return v;
}

void removeString(std::string& str, const std::string& substr) {
    std::string::size_type pos = 0;

    while ((pos = str.find(substr, pos)) != std::string::npos)
        str.erase(pos, substr.length());
}

static inline uint blendPixels(uint pixel1, uint pixel2)
{
    uchar alpha1 = (pixel1 >> 24) & 0xFF;
    uchar alpha2 = (pixel2 >> 24) & 0xFF;

    float normAlpha1 = alpha1 / 255.0f;
    float normAlpha2 = alpha2 / 255.0f;

    float newAlpha = normAlpha1 + (1 - normAlpha1) * normAlpha2;

    uchar red1 = (pixel1 >> 16) & 0xFF;
    uchar red2 = (pixel2 >> 16) & 0xFF;
    uchar green1 = (pixel1 >> 8) & 0xFF;
    uchar green2 = (pixel2 >> 8) & 0xFF;
    uchar blue1 = pixel1 & 0xFF;
    uchar blue2 = pixel2 & 0xFF;

    uchar newRed = static_cast<uchar>((red1 * normAlpha1 + red2 * normAlpha2 * (1 - normAlpha1)) / newAlpha);
    uchar newGreen = static_cast<uchar>((green1 * normAlpha1 + green2 * normAlpha2 * (1 - normAlpha1)) / newAlpha);
    uchar newBlue = static_cast<uchar>((blue1 * normAlpha1 + blue2 * normAlpha2 * (1 - normAlpha1)) / newAlpha);

    uint blendedPixel = (static_cast<uint>(newAlpha * 255) << 24) | (newRed << 16) | (newGreen << 8) | newBlue;

    return blendedPixel;
}

static void blit(uint* dest, const uint* src, uint dWidth, uint dHeight, uint sWidth, uint sHeight, uint centerX, uint centerY, uint color) {
    uint startX = centerX - sWidth / 2;
    uint startY = centerY - sHeight / 2;

    for (uint sy = sHeight - 1; sy != (uint)-1; sy--)
    {
        for (uint sx = 0; sx < sWidth; sx++)
        {
            uint dx = startX + sx;
            uint dy = startY + (sHeight - 1 - sy);

            // Check if the destination coordinates are within the bounds of the destination buffer
            if (dx >= 0 && dx < dWidth && dy >= 0 && dy < dHeight)
            {
                uint srcCol = src[sy * sWidth + sx];
                uint destIdx = dy * dWidth + dx;

                // Extract the alpha channel from the source pixel
                unsigned char alpha = (srcCol >> 24) & 0xFF;

                if (alpha > 250) // If the source pixel is opaque, overwrite the destination pixel
                    dest[destIdx] = (srcCol & 0xff000000) | (color & 0x00ffffff);
                else if (alpha == 0x00) // If the source pixel is transparent, skip
                    continue;
                else { // If the source pixel is semi-transparent, blend the source and destination pixels
                    dest[destIdx] = blendPixels(dest[destIdx], srcCol);
                }
            }
        }
    }
}

// Recolor bitmap while preserving alpha channel
void recolorBitmap(uint* pixels, uint width, uint height, uint color) {
    const uint size = width * height;

    for (uint i = 0; i < size; i++)
        pixels[i] = (pixels[i] & 0xff000000) | (color & 0x00ffffff);
}

static vector<string> markerNames;        // Marker names for lookup
static vector<string> markerNamesSafe;    // Names without "Markers/" and ".svg"
static vector<string> markerNamesSafeIDs; // IDs for ImGui
static vector<uint>   markerPreviews;     // Rasterized marker previews
static vector<string> markerData;         // Marker svg data
static uint           markerCount;        // Number of markers

void InitPlotFieldRenderers() {
    markerNames = getMapKeys(markers);

    for (const auto& name : markerNames) {
        markerData.push_back(markers[name]);

        string safeName = name;
        
        removeString(safeName, "Markers/");
        removeString(safeName, ".svg");

        markerNamesSafe.push_back(safeName);
        markerNamesSafeIDs.push_back("##" + safeName);

        // Build marker previews
        auto pixels = LoadSVGPixels(markers[name].c_str(), 16, 16, 1.0f);
        recolorBitmap(pixels, 16, 16, 0xFFE3E3E3);
        markerPreviews.push_back(uploadTexture(pixels, 16, 16));

        delete[] pixels;
    }

    markerCount = markerNames.size();
}

template <typename T>
T* rotateCC(const T* data, int width, int height) {
    T* buf = nullptr;
    try {
        buf = new T[width * height];
    } catch (...) {
        DispError("RendererSpectra::RenderPlot::rotateCC", "Failed to allocate texture memory!");
        return nullptr;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            buf[j * height + i] = data[i * width + (width - 1) - j];

    return buf;
}


static float luma(uint color) {
    uchar red   = (color >> 16) & 0xFF;
    uchar green = (color >> 8)  & 0xFF;
    uchar blue  =  color        & 0xFF;

    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Crop / scale spectra plot
static uint* getBox(const uint* pixels, uint width, uint height, Birch::Point<float> inputTL, Birch::Point<float> inputBR, Birch::Point<float> outputTL, Birch::Point<float> outputBR, uint outputWidth, uint outputHeight) {
    // Ensure that the input and output coordinates are within the image boundaries
    inputTL.X = fmax(0, fmin(inputTL.X, width  - 1));
    inputTL.Y = fmax(0, fmin(inputTL.Y, height - 1));
    inputBR.X = fmax(0, fmin(inputBR.X, width  - 1));
    inputBR.Y = fmax(0, fmin(inputBR.Y, height - 1));

    outputTL.X = fmax(0, fmin(outputTL.X, outputWidth  - 1));
    outputTL.Y = fmax(0, fmin(outputTL.Y, outputHeight - 1));
    outputBR.X = fmax(0, fmin(outputBR.X, outputWidth  - 1));
    outputBR.Y = fmax(0, fmin(outputBR.Y, outputHeight - 1));

    // Calculate the crop dimensions
    int cropWidth  = static_cast<int>(outputBR.X - outputTL.X);
    int cropHeight = static_cast<int>(outputBR.Y - outputTL.Y);

    uint* scaledPixels = new uint[outputWidth * outputHeight];

    // Iterate through the pixels of the scaled image
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            // Map the coordinates from output space to input space
            float inputX = inputTL.X + (x / static_cast<float>(outputWidth))  * (inputBR.X - inputTL.X);
            float inputY = inputTL.X + (y / static_cast<float>(outputHeight)) * (inputBR.Y - inputTL.Y);

            // Round the input coordinates to the nearest integer
            int inputX_int = static_cast<int>(inputX);
            int inputY_int = static_cast<int>(inputY);

            // Map the 2D coordinates to a 1D index
            int inputIndex = inputY_int * width + inputX_int;

            // Map the coordinates to the scaled image
            int scaledIndex = y * outputWidth + x;

            // Find the maximum color value in the input region
            float maxColor = 0;
            for (int dy = 0; dy < cropHeight; ++dy) {
                for (int dx = 0; dx < cropWidth; ++dx) {
                    int inputPixelIndex = (inputY_int + dy) * width + (inputX_int + dx);
                    uint inputColor = pixels[inputPixelIndex];
                    maxColor = std::max(maxColor, luma(inputColor));
                }
            }

            scaledPixels[scaledIndex] = maxColor;
        }
    }

    return scaledPixels;
}

static GLuint rasterizeSpectraPlot(const unsigned* pixels, int width, int height) {
    //auto timer = Stopwatch();
    const uint maxOutRes = 4096;

    // Crop / scale the image to the current plot size
    //uint* cropBuf = getBox(pixels, width, height, inputTL, inputBR, outputTL, outputBR, maxOutRes, maxOutRes);

    #ifdef FFT_IMPL_CUDA
        auto buf = rotateCCCUDA(pixels, width, height);
    #else
        auto buf = rotateCC(pixels, width, height);
    #endif

    //delete[] cropBuf;

    //DispDebug("Rotation time: %lf", timer.Now());

    if (buf == nullptr) {
        DispError("RendererSpectra::RenderPlot::rasterizeSpectraPlot", "Texture buffer was null! Plot will not render!");
        return 0;
    }

    GLuint tex;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glActiveTexture(tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, height, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf);

    delete[] buf;

    return tex;
}
static void discardTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = 0;
}

static bool markerSelect(const char* label, uint* marker, uint* hoveredMarker, bool* isHovered) {
    bool changed = false;

    if (ImGui::BeginCombo(label, markerNamesSafe[*marker].c_str())) {
        for (uint i = 0; i < markerCount; i++) {
            if (ImGui::Selectable(markerNamesSafeIDs[i].c_str(), i == *marker)) {
                *marker = i;
                changed = true;
            }

            if (ImGui::IsItemHovered()) {
                *hoveredMarker = i;
                *isHovered = true;
            }

            ImGui::SameLine();
            ImGui::Image(t_ImTexID(markerPreviews[i]), ImVec2(16, 16));
            ImGui::SameLine();
            ImGui::Text("%s", markerNamesSafe[i].c_str());
        }

        ImGui::EndCombo();
    }

    return changed;
}

PlotFieldRenderer::PlotFieldRenderer(PlotField* field) : field(field) { }
bool PlotFieldRenderer::AutoFit() { return autofit; }
void PlotFieldRenderer::SetAutoFit(bool fit) { autofit = fit; }

const char* RendererShaded::Name()  { return "Shaded"; }
const char* RendererLine::Name()    { return "Line"; }
const char* RendererScatter::Name() { return "Scatter"; }
const char* RendererSpectra::Name() { return "Spectra"; }

RendererShaded::RendererShaded(PlotField* field) : PlotFieldRenderer(field) { }
RendererLine::RendererLine(PlotField* field) : PlotFieldRenderer(field) { }
RendererScatter::RendererScatter(PlotField* field, bool zmod, ScatterStyle style) : PlotFieldRenderer(field), zmod(zmod), style(style) { }
RendererSpectra::RendererSpectra(PlotField* field) : PlotFieldRenderer(field) { }

void RendererShaded::RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) {
    bool* volatile mask = nullptr;
    if (field->UseFilter != nullptr && *field->UseFilter)
        mask = field->FilterMask;

    const auto elCount    = *field->ElementCount;
    const bool drawPoints = elCount < pointThres;

    if (drawPoints) {
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker,     marker);
        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, markerSize);
    }

/*
    static uint test = 0;

    if (!test) {
        // Load image.png
        int w, h, n;
        auto pixels = stbi_load("image.png", &w, &h, &n, 4);

        if (pixels == nullptr) {
            DispError("RendererShaded::RenderPlot", "Failed to load image.png!");
            return;
        }

        // Create texture
        test = rasterizeImagePlot((unsigned*)pixels, w, h);

        // Free image
        stbi_image_free(pixels);
    }

    auto plotLimits = ImPlot::GetPlotLimits();

    ImPlotPoint sD = ImPlotPoint(plotLimits.X.Min, field->YLimits.End);
    ImPlotPoint eD = ImPlotPoint(plotLimits.X.Max, field->YLimits.Start);

    ImPlot::PlotImage("##Image", t_ImTexID(test), sD, eD);
*/
    ImPlot::PlotShaded(field->Name, field->Timing, field->Data, field->ShadedColormap, opacity, mask, elCount);
    ImPlot::PlotLine(field->Name,   field->Timing, field->Data, field->Colormap,                mask, elCount);

    if (drawPoints)
        ImPlot::PopStyleVar(2);
}
bool RendererShaded::RenderWidget() {
    bool changed = false;

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::SliderFloat("Marker Size", &markerSize, 1.0f, 50.0f, "Size: %.2f");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::Combo("Marker", (int*)&marker, "None\0Circle\0Square\0Diamond\0Up\0Down\0Left\0Right\0Plus\0Cross\0Asterisk\0");

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::SliderFloat("Fill", &opacity, 0.0f, 1.0f, "Opacity: %.2f");

    return changed;
}

void RendererLine::RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) {
    const bool drawPoints = field->ViewElementCount < pointThres;
    const bool useFilter  = field->UseFilter == nullptr ? false : *field->UseFilter;

    if (drawPoints) {
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, marker);
        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, markerSize);
    }

     ImPlot::PlotLine(field->Name, field->Timing, field->Data, field->Colormap, useFilter? field->FilterMask : nullptr, *field->ElementCount);

    if (drawPoints)
        ImPlot::PopStyleVar(2);
}
bool RendererLine::RenderWidget() {
    bool changed = false;

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::SliderFloat("Marker Size", &markerSize, 1.0f, 50.0f);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::Combo("Marker", (int*)&marker, "None\0Circle\0Square\0Diamond\0Up\0Down\0Left\0Right\0Plus\0Cross\0Asterisk\0");

    return changed;
}

uint colormapMode(const uint* colors, uint len) {
    uint maxFreq = 0, modeColor = 0;

    #pragma omp parallel
    {
        std::unordered_map<uint, uint> freqMap;
        uint tmMaxFreq = 0, tempMode = 0;

        #pragma omp for
        for (uint i = 0; i < len; i++) {
            freqMap[colors[i]]++;
            if (freqMap[colors[i]] > tmMaxFreq) {
                tmMaxFreq = freqMap[colors[i]];
                tempMode = colors[i];
            }
        }

        #pragma omp critical
        {
            if (tmMaxFreq > maxFreq) {
                maxFreq = tmMaxFreq;
                modeColor = tempMode;
            }
        }
    }

    return modeColor;
}

void renderPoints(const PlotField* const field, const Span<double>& xLimits, const Span<double>& yLimits, int xRes, int yRes, bool useFilter, bool* renderMask, uint* pixels, const uint* color) {
    const double xScale = (xRes - 1) / (xLimits.End - xLimits.Start);
    const double yScale = (yRes - 1) / (yLimits.End - yLimits.Start);

    const ulong count = *field->ElementCount;

    const double* const xVals = field->Timing;
    const double* const yVals = field->Data;
    const bool*   const filt  = field->FilterMask;
    const uint*   const col   = field->Colormap;

    const uint mode = colormapMode(col, count);

    #pragma omp parallel for
    for (ulong i = 0; i < count; i++) {
        if (!yLimits.Contains(yVals[i]))
            continue;
        if (!xLimits.Contains(xVals[i]))
            continue;
        if (useFilter && !filt[i])
            continue;

        const auto pos = Point<unsigned>((xVals[i] - xLimits.Start) * xScale, (yVals[i] - yLimits.Start) * yScale);
        const auto gridIdx = pos.Y * xRes + pos.X;

        if (gridIdx >= xRes * yRes) {
            DispError("renderPoints", "Grid index out of bounds!\n\tIdx: %d\n\tgDim: %dx%d\n\tgPos: %lf, %lf\n\tpPos: %lf, %lf\n\txLims: %lf -> %lf\n\tyLims: %lf -> %lf", gridIdx, xRes, yRes, pos.X, pos.Y, xVals[i], yVals[i], xLimits.Start, xLimits.End, yLimits.Start, yLimits.End);
            continue;
        }

        if (color == nullptr)
        {
            if (pixels[gridIdx] == 0)
                pixels[gridIdx] = col[i];
            else if (pixels[gridIdx] == mode)
                pixels[gridIdx] = col[i];
        }
        else
        {
            pixels[gridIdx] = *color;
        }
    }
}
void renderMarkers(const PlotField* const field, const Span<double>& xLimits, const Span<double>& yLimits, int xRes, int yRes, bool useFilter, bool* renderMask, uint* pixels, uint* markerPixels, uint markerSize, const uint* color) {
    const double xScale = xRes / (xLimits.End - xLimits.Start);
    const double yScale = yRes / (yLimits.End - yLimits.Start);

    const ulong count = *field->ElementCount;

    const double* const xVals = field->Timing;
    const double* const yVals = field->Data;
    const bool*   const filt  = field->FilterMask;
    const uint*   const col   = field->Colormap;

    #ifdef FFT_IMPL_CUDA
        uint* pixels_cu       = copyToCUDA(pixels, xRes * yRes);
        uint* markerPixels_cu = copyToCUDA(markerPixels, markerSize * markerSize);
    #endif

    for (ulong i = 0; i < count; i++) {
        if (!yLimits.Contains(yVals[i]))
            continue;
        if (!xLimits.Contains(xVals[i]))
            continue;
        if (useFilter && !filt[i])
            continue;

        const auto pos = Point<unsigned>((xVals[i] - xLimits.Start) * xScale, (yVals[i] - yLimits.Start) * yScale);
        const auto gridIdx = pos.Y * xRes + pos.X;

        if (renderMask[gridIdx])
            continue;

        renderMask[gridIdx] = true;
/*
        #ifdef FFT_IMPL_CUDA
            if (color == nullptr)
                blitCUDA(pixels_cu, markerPixels_cu, xRes, yRes, markerSize, markerSize, pos.X, pos.Y, col[i]);
            else
                blitCUDA(pixels_cu, markerPixels_cu, xRes, yRes, markerSize, markerSize, pos.X, pos.Y, *color);
        #else
        */
            if (color == nullptr)
                blit(pixels, markerPixels, xRes, yRes, markerSize, markerSize, pos.X, pos.Y, col[i]);
            else
                blit(pixels, markerPixels, xRes, yRes, markerSize, markerSize, pos.X, pos.Y, *color);
        //#endif
    }

    #ifdef FFT_IMPL_CUDA
        syncCUDA();

        copyFromCUDA(pixels_cu, pixels, xRes * yRes);

        freeCUDA(pixels_cu);
        freeCUDA(markerPixels_cu);
    #endif
}
void renderStems(const PlotField* const field, const Span<double>& xLimits, const Span<double>& yLimits, int xRes, int yRes, bool useFilter, bool* renderMask, uint* pixels, const uint* color) {
    const double xScale = xRes / (xLimits.End - xLimits.Start);
    const double yScale = yRes / (yLimits.End - yLimits.Start);

    const ulong count = *field->ElementCount;

    const double* const xVals = field->Timing;
    const double* const yVals = field->Data;
    const bool*   const filt  = field->FilterMask;
    const uint*   const col   = field->Colormap;

    #pragma omp parallel for
    for (ulong i = 0; i < count; i++) {
        if (yLimits.Start > yVals[i])
            continue;
        if (!xLimits.Contains(xVals[i]))
            continue;
        if (useFilter && !filt[i])
            continue;
        
        const bool offScreen = yLimits.End <= yVals[i];

        const auto pos = Point<unsigned>((xVals[i] - xLimits.Start) * xScale, ((offScreen ? yLimits.End : yVals[i]) - yLimits.Start) * yScale);
        const auto gridIdx = pos.Y * xRes + pos.X;

        for (uint j = 0; j < pos.Y; j++) {
            const auto idx = j * xRes + pos.X;

            renderMask[idx] = true;

            if (color == nullptr)
                pixels[idx] = col[i];
            else
                pixels[idx] = *color;
        }
    }
}

void renderStairs(const PlotField* const field, const Span<double>& xLimits, const Span<double>& yLimits, int xRes, int yRes, bool useFilter, bool* renderMask, uint* pixels) {
    const double xScale = xRes / (xLimits.End - xLimits.Start);
    const double yScale = yRes / (yLimits.End - yLimits.Start);

    const ulong count = *field->ElementCount;

    const double* const xVals = field->Timing;
    const double* const yVals = field->Data;
    const bool*   const filt  = field->FilterMask;
    const uint*   const col   = field->Colormap;

    #pragma omp parallel for
    for (ulong i = 1; i < count; i++) {
        if (!yLimits.Contains(yVals[i]))
            continue;
        if (!xLimits.Contains(xVals[i]))
            continue;
        if (useFilter && !filt[i])
            continue;
        if (i == count -1)
            continue;

        const auto pos = Point<unsigned>((xVals[i - 1] - xLimits.Start) * xScale, (yVals[i - 1] - yLimits.Start) * yScale);
        const auto nextPos = Point<unsigned>((xVals[i] - xLimits.Start) * xScale, (yVals[i] - yLimits.Start) * yScale);

        // Horizontal line
        for (uint x = pos.X; x < nextPos.X; x++)
            pixels[pos.Y * xRes + x]    = col[i];
        // Vertical line
        for (uint y = pos.Y; y < nextPos.Y; y++)
            pixels[y * xRes + nextPos.X] = col[i];
    }
}

GLuint renderScatterPlot(const PlotField* const field, const Span<double>& xLimits, const Span<double>& yLimits, int xRes, int yRes, bool useFilter, ScatterStyle style, uint marker, uint markerSize, const uint* color, uint pointThres) {
    // Setup
    GLuint texture = 0;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (field == nullptr || field->ElementCount == nullptr || *field->ElementCount == 0)
        return texture;

    bool* renderMask = new bool[xRes * yRes];
    std::fill_n(renderMask, xRes * yRes, false);

    uint* pixels = new uint[xRes * yRes];
    std::fill_n(pixels, xRes * yRes, 0x00000000);

    // Render
    switch (style) {
    case ScatterStyle::Points:
        if (markerSize == 1) {
            //#ifdef FFT_IMPL_CUDA
            //    renderPointsCUDA(field->Data, field->Timing, field->FilterMask, field->Colormap, *field->ElementCount, xLimits.Start, xLimits.End, yLimits.Start, yLimits.End, xRes, yRes, useFilter, renderMask, pixels);
            //#else
                renderPoints(field, xLimits, yLimits, xRes, yRes, useFilter, renderMask, pixels, color);
            //#endif
        }
        else {
            uint* markerPix = LoadSVGPixels(markerData[marker].c_str(), markerSize, markerSize, 1.0f);

            renderMarkers(field, xLimits, yLimits, xRes, yRes, useFilter, renderMask, pixels, markerPix, markerSize, color);

            delete[] markerPix;
        }
        break;
    case ScatterStyle::Stems:
        renderStems(field, xLimits, yLimits, xRes, yRes, useFilter, renderMask, pixels, color);
        if (markerSize != 1) {
            uint* markerPix = LoadSVGPixels(markerData[marker].c_str(), markerSize, markerSize, 1.0f);

            renderMarkers(field, xLimits, yLimits, xRes, yRes, useFilter, renderMask, pixels, markerPix, markerSize, color);

            delete[] markerPix;
        }
        break;
    case ScatterStyle::Lines:
        break;
    case ScatterStyle::Stairs:
        renderStairs(field, xLimits, yLimits, xRes, yRes, useFilter, renderMask, pixels);
        break;
    };



    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, xRes, yRes, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    // Cleanup
    delete[] pixels;
    delete[] renderMask;

    return texture;
}

void RendererScatter::RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) {
    bool* volatile filterMask = field->FilterMask;
    if (ImGui::IsKeyDown(ImGuiKey_LeftShift) && !strcmp(currentTool->Name(), "Filter")) {
        filterMask = nullptr;
        shiftDown = true;
    } else {
        shiftDown = false;
    }

    bool useFilter = filterMask != nullptr;

    auto plotLimits = ImPlot::GetPlotLimits();

    ImPlotPoint* sD = &cont->TextureDimensions[0];
    ImPlotPoint* eD = &cont->TextureDimensions[1];

    if (ImPlot::FitThisFrame())
        cont->Rasterized = false;

    if (field->DataVersion != lastVer || !cont->Rasterized || (shiftDown != lastShiftDown) || styleChanged) {
        discardTexture(&cont->Texture);

        if (ImPlot::FitThisFrame()) {
            cont->TextureDimensions[0] = ImPlotPoint(plotLimits.X.Min, field->YLimits.End);
            cont->TextureDimensions[1] = ImPlotPoint(plotLimits.X.Max, field->YLimits.Start);
        } else {
            cont->TextureDimensions[0] = ImPlotPoint(plotLimits.X.Min, plotLimits.Y.Max);
            cont->TextureDimensions[1] = ImPlotPoint(plotLimits.X.Max, plotLimits.Y.Min);
        }

        auto plotRect  = ImPlot::GetCurrentPlot()->PlotRect;
        ImVec2 fboPos  = ImVec2(plotRect.Min.x, plotRect.Min.y);
        ImVec2 fboSize = ImVec2(plotRect.GetWidth(), plotRect.GetHeight());

        // Convert float color to uint
        uint staticColor = 0;
        for (int i = 0; i < 4; i++) {
            uint8_t comp = static_cast<uint8_t>(color[i] * 255.0f);
            staticColor |= (comp << (8 * i));
        }

        auto timer = Stopwatch();
        cont->Texture = renderScatterPlot(field, Span<double>(plotLimits.X.Min, plotLimits.X.Max), Span<double>(plotLimits.Y.Min, plotLimits.Y.Max), fboSize.x, fboSize.y, useFilter, style, marker, markerSize, zmod ? nullptr : &staticColor, pointThres);
        //DispDebug("RenderScatterPlot in %lfs", timer.Now());
        timer.Reset();

        lastVer = field->DataVersion;
        cont->Rasterized = true;

        cont->TextureSize = fboSize;

        styleChanged = false;
    }

    ImVec2 p1 = ImPlot::PlotToPixels(cont->TextureDimensions[0].x, cont->TextureDimensions[1].y);
    ImVec2 p2 = ImPlot::PlotToPixels(cont->TextureDimensions[1].x, cont->TextureDimensions[0].y);

    cont->LastRenderedSize = ImVec2(p2.x - p1.x, p2.y - p1.y);

    auto dl = ImPlot::GetPlotDrawList();
    dl->AddCallback([](const ImDrawList* parent_list, const ImDrawCmd* cmd) -> void {
        PlotFieldContainer* cont = (PlotFieldContainer*)cmd->UserCallbackData;

        ImDrawData* draw_data = ImGui::GetDrawData();
        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

        const float ortho_projection[] =
        {
            2.0f/(R-L),   0.0f,         0.0f,   0.0f,
            0.0f,         2.0f/(T-B),   0.0f,   0.0f,
            0.0f,         0.0f,        -1.0f,   0.0f,
            (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f
        };

        gShaders["scatter"]->Use();
        gShaders["scatter"]->SetMatrix4("projection", ortho_projection);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cont->Texture);

        gShaders["scatter"]->SetSampler2D("tex", 0);
        gShaders["scatter"]->SetVector2("texSize",    cont->TextureSize);
        gShaders["scatter"]->SetVector2("renderSize", cont->LastRenderedSize);
    }, cont);
    ImPlot::PlotImage(field->Name, t_ImTexID(cont->Texture), *sD, *eD);

    dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

    lastShiftDown = shiftDown;
}
bool RendererScatter::RenderWidget() {
    bool changed = false;

    if (lastHover) {
        marker = lastMarker;
        lastHover = false;
    }

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::Combo("Style", (int*)&style, "Points\0Stems\0");

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    int ms = (int)markerSize;
    changed |= ImGui::DragInt("Marker Size", &ms, 1, 1, 50, "Size: %d");
    markerSize = (uint)ms;

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    bool isHovered = false;
    uint hoveredMarker = 0;    
    if (!markerSelect("Marker", &marker, &hoveredMarker, &isHovered)) {
        if (isHovered) {
            lastMarker = marker;
            marker = hoveredMarker;

            lastHover = true;
        }
        
        changed = true;
    } else {
        changed = true;
    }

    changed |= ImGui::Checkbox("Use Colormap", &zmod);
    if (!zmod) {
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);

        changed |= ImGui::ColorEdit4("Color", color, ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs);
    }

    styleChanged |= changed;

    return changed;
}

static inline bool fEq(float a, float b, float e) {
    return fabsf(a - b) <= e;
}

void RendererSpectra::RenderPlot(const Tool* currentTool, PlotFieldContainer* cont) {
    cont->TextureDimensions[0] = ImPlotPoint((field->XLimits.Start - field->XOffset.Start), (field->YLimits.Start - field->YOffset.Start));
    cont->TextureDimensions[1] = ImPlotPoint((field->XLimits.End   + field->XOffset.End),   (field->YLimits.End   + field->YOffset.Start));

    ImPlotPoint* sD = &cont->TextureDimensions[0];
    ImPlotPoint* eD = &cont->TextureDimensions[1];

    if (cont->LastTextureDimensions[0] == nullptr) {
        cont->LastTextureDimensions[0] = new ImPlotPoint(sD->x, sD->y);
        cont->LastTextureDimensions[1] = new ImPlotPoint(eD->x, eD->y);

        cont->Rasterized = false;
    }

    if (!fEq(sD->x, cont->LastTextureDimensions[0]->x, .00001f) || !fEq(eD->x, cont->LastTextureDimensions[1]->x, .00001f) || 
        !fEq(sD->y, cont->LastTextureDimensions[0]->y, .00001f) || !fEq(eD->y, cont->LastTextureDimensions[1]->y, .00001f)) {
        cont->Rasterized = false;
    }

    auto plotLimits = ImPlot::GetPlotLimits();

    if (field->DataVersion != lastVer || !cont->Rasterized) {
        if (field->ColormapReady) {
            discardTexture(&cont->Texture);
            cont->Texture = rasterizeSpectraPlot(field->Colormap, field->XRes, field->YRes);
            cont->Rasterized = true;
            cont->TextureSize = ImVec2(field->YRes, field->XRes); // These are flipped because the texture is rotated 90 degrees

            //printf("rendered\n");

            delete cont->LastTextureDimensions[0];
            delete cont->LastTextureDimensions[1];

            cont->LastTextureDimensions[0] = new ImPlotPoint(sD->x, sD->y);
            cont->LastTextureDimensions[1] = new ImPlotPoint(eD->x, eD->y);

            lastVer = field->DataVersion;
        } else {
            sD = cont->LastTextureDimensions[0];
            eD = cont->LastTextureDimensions[1];
        }
    }

    ImVec2 p1 = ImPlot::PlotToPixels(cont->TextureDimensions[0].x, cont->TextureDimensions[1].y);
    ImVec2 p2 = ImPlot::PlotToPixels(cont->TextureDimensions[1].x, cont->TextureDimensions[0].y);

    cont->LastRenderedSize = ImVec2(p2.x - p1.x, p2.y - p1.y);

    auto dl = ImPlot::GetPlotDrawList();
    dl->AddCallback([](const ImDrawList* parent_list, const ImDrawCmd* cmd) -> void {
        PlotFieldContainer* cont = (PlotFieldContainer*)cmd->UserCallbackData;

        ImDrawData* draw_data = ImGui::GetDrawData();
        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

        const float ortho_projection[] =
        {
            2.0f/(R-L),   0.0f,         0.0f,   0.0f,
            0.0f,         2.0f/(T-B),   0.0f,   0.0f,
            0.0f,         0.0f,        -1.0f,   0.0f,
            (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f
        };

        gShaders["spectra"]->Use();
        gShaders["spectra"]->SetMatrix4("projection", ortho_projection);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cont->Texture);

        gShaders["spectra"]->SetSampler2D("tex", 0);
        gShaders["spectra"]->SetVector2("texSize",    cont->TextureSize);
        gShaders["spectra"]->SetVector2("renderSize", cont->LastRenderedSize);
    }, cont);

    ImPlot::PlotImage(field->Name, t_ImTexID(cont->Texture), *sD, *eD);

    dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}
bool RendererSpectra::RenderWidget() {
    bool changed = false;

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);
    ImGui::Text("FFT Size:");
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);

    if (ImGui::Combo("##fftsize", &fftSize, sizes, 6)) {
        datSize = FFTSize();
        overlap = 0;

        changed = true;
    }

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);
    ImGui::Text("Window:");
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= WindowWidget(&window, GetWindows());

    ImGui::PushID("datasel");

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 30);
    if (ImGui::DragInt("##datasize", &datSize, 1, 2, FFTSize(), "Data: %d")) {
        overlap = 0;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("%"))
        ImGui::OpenPopup("##dataSel");
    
    if (ImGui::BeginPopup("##dataSel")) {
        if (ImGui::MenuItem("50%")) {
            datSize = (int)(FFTSize() * 0.5f);
            overlap = 0;
            changed = true;
        }

        if (ImGui::MenuItem("100%")) {
            datSize = FFTSize();
            overlap = 0;
            changed = true;
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();
    ImGui::PushID("overlapsel");

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 30);
    changed |= ImGui::DragInt("##overlap", &overlap, 1, 0, datSize - 1, "Overlap: %d");
    ImGui::SameLine();
    if (ImGui::Button("%"))
        ImGui::OpenPopup("##overlapSel");
    
    if (ImGui::BeginPopup("##overlapSel")) {
        if (ImGui::MenuItem("0%")) {
            overlap = 0;
            changed = true;
        }

        if (ImGui::MenuItem("25%")) {
            overlap = (int)(datSize * 0.25f);
            changed = true;
        }

        if (ImGui::MenuItem("50%")) {
            overlap = (int)(datSize * 0.5f);
            changed = true;
        }

        if (ImGui::MenuItem("75%")) {
            overlap = (int)(datSize * 0.75f);
            changed = true;
        }

        ImGui::EndPopup();
    }

    ImGui::PopID();

    if (changed)
        field->NeedsUpdate = true;

    return changed;
}
int RendererSpectra::FFTSize() { return pow(2, fftSize + 3); }
int RendererSpectra::TexSize() { return FFTSize(); }
int RendererSpectra::Overlap() { return overlap; }
int RendererSpectra::DatSize() { return datSize; }
WindowFunc* RendererSpectra::Window() { return window; }
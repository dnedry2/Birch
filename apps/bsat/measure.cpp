#include <math.h>
#include <iostream>
#include <algorithm>

#include "measure.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using namespace Birch;

Measurement::Measurement(const Birch::Point<double>& position, const Birch::Span<double>& bounds, PlotField* field, bool screen, llong dataIdx, MeasureDisplay displayMode, char* xUnit, char* yUnit)
           : position(position), bounds(bounds), displayMode(displayMode), screen(screen), field(field), dataIdx(dataIdx), xUnit(xUnit), yUnit(yUnit)
{

}
Measurement::~Measurement()
{

}

PlotField* Measurement::Field()
{
    return field;
}

Birch::Point<double>* Measurement::Position()
{
    return &position;
}

Birch::Span<double>* Measurement::Bounds()
{
    return &bounds;
}

MeasureDisplay* Measurement::DisplayMode()
{
    return &displayMode;
}

bool* Measurement::Screen()
{
    return &screen;
}

llong* Measurement::DataIdx()
{
    return &dataIdx;
}

char* Measurement::XUnit()
{
    return xUnit;
}
char* Measurement::YUnit()
{
    return yUnit;
}
double* Measurement::XOffset()
{
    return &xOffset;
}
double* Measurement::YOffset()
{
    return &yOffset;
}
std::string* Measurement::AltText()
{
    return &altText;
}


HistMeasurement::HistMeasurement(const Birch::Span<double>& bounds, PlotField* field)
              : bounds(bounds), field(field) { }

PlotField* HistMeasurement::Field()
{
    return field;
}
Birch::Span<double>* HistMeasurement::Bounds()
{
    return &bounds;
}
float* HistMeasurement::Curve()
{
    return curve;
}
char*& HistMeasurement::Stats()
{
    return stats;
}


// Only need to determine which point is closer, so don't calculate the actual distance
static inline double distScore(double x1, double y1, double x2, double y2) {
    #pragma float_control(precise, off)
    return pow(x2 - x1, 2) + pow(y2 - y1, 2);
}

// Finds the nearest value in a sorted array
static inline uint findNearest(const double* input, uint len, double needle) {
    auto it = std::lower_bound(input, input + len, needle);
    return std::distance(input, std::min(it, input + len - 1));
}

// Find closest point in visual space
static uint sSearch(const double* x, const double* y, const bool* mask, uint startIdx, uint endIdx, const Span<double>& yLimits, const ImPlotPoint& mousePix) {
    uint   closestIdx  = startIdx;
    double closestDist = std::numeric_limits<double>::max();

    #pragma omp parallel shared(closestDist, closestIdx)
    {
        uint   localClosestIdx  = startIdx;
        double localClosestDist = std::numeric_limits<double>::max();

        #pragma omp for
        for (uint i = startIdx; i < endIdx; ++i) {
            // Ignore filtered points and points outside of box
            if (mask[i] && yLimits.Contains(y[i])) {
                const auto canPix  = ImPlot::PlotToPixels(x[i], y[i]);
                const auto canDist = distScore(mousePix.x, mousePix.y, canPix.x, canPix.y);

                if (canDist < localClosestDist) {
                    localClosestDist = canDist;
                    localClosestIdx  = i;
                }
            }
        }

        #pragma omp critical
        {
            if (localClosestDist < closestDist) {
                closestDist = localClosestDist;
                closestIdx  = localClosestIdx;
            }
        }
    }

    return closestIdx;
}

uint SmartSearch(const double* x, const double* y, const bool* mask, uint count, const Span<double>& yLimits, const ImPlotPoint& mousePos) {
    // Constants
    const auto mousePix = ImPlot::PlotToPixels(mousePos);
    const auto plotSize = ImPlot::GetPlotSize();


    // Calculate max x distance to search based on the max y distance
    const int    maxYDist = plotSize.y - mousePix.y > mousePix.y ? plotSize.y - mousePix.y : mousePix.y;
    const double maxDistX = ImPlot::PixelsToPlot(ImVec2(mousePix.x + maxYDist, 0)).x;


    // Get start and end index
    uint startIdx = findNearest(x, count, mousePos.x);
    uint endIdx   = findNearest(x, count, maxDistX);


    return sSearch(x, y, mask, startIdx, endIdx, yLimits, mousePix);
}

uint SmartSearchBox(const double* x, const double* y, const bool* mask, uint count, const ImPlotPoint& mousePos, uint pixTolerance) {
    // Constants
    const auto mousePix = ImPlot::PlotToPixels(mousePos);
    const auto plotLims = ImPlot::GetPlotLimits();


    // Calculate max x / y distance to search based on tolerance
    auto xPixLimits = Birch::Span<double>(mousePix.x - pixTolerance / 2, mousePix.x + pixTolerance / 2);
    auto yPixLimits = Birch::Span<double>(mousePix.y - pixTolerance / 2, mousePix.y + pixTolerance / 2);

    if (xPixLimits.Start > xPixLimits.End)
        std::swap(xPixLimits.Start, xPixLimits.End);
    if (yPixLimits.Start < yPixLimits.End)
        std::swap(yPixLimits.Start, yPixLimits.End);

    auto xLimits = Birch::Span<double>(ImPlot::PixelsToPlot(ImVec2(xPixLimits.Start, 0)).x, ImPlot::PixelsToPlot(ImVec2(xPixLimits.End, 0)).x);
    auto yLimits = Birch::Span<double>(ImPlot::PixelsToPlot(ImVec2(0, yPixLimits.Start)).y, ImPlot::PixelsToPlot(ImVec2(0, yPixLimits.End)).y);

    xLimits.Clamp(plotLims.X.Min, plotLims.X.Max);
    yLimits.Clamp(plotLims.Y.Min, plotLims.Y.Max);


    // Get start and end index
    uint startIdx = findNearest(x, count, xLimits.Start);
    uint endIdx   = findNearest(x, count, xLimits.End);


    return sSearch(x, y, mask, startIdx, endIdx, yLimits, mousePix);
}

Birch::Point<double> SmartSearchSpectra(const double* data, uint xRes, uint yRes, const Birch::Point<double>& mousePos, Birch::Span<double> dataXLimits, Birch::Span<double> dataYLimits, double maxValue) {
    // Image is rotated 90deg CW at this point, which is why x and y are swapped at some points

    // Constants
    const auto plotLims  = ImPlot::GetPlotLimits();
    const auto pixScaleX = yRes / dataXLimits.Length();
    const auto pixScaleY = xRes / dataYLimits.Length();
    const auto framePos  = ImPlot::GetPlotPos();
    const auto frameSize = ImPlot::GetPlotSize();
    const auto mousePos3 = Birch::Point3<double>(mousePos.X, mousePos.Y, sqrt(pow(dataXLimits.Length(), 2) + pow(dataYLimits.Length(), 2)));

    // Power needs to be scaled depending on the texture size
    const auto pwrScale = mousePos3.Z / log(maxValue);

    // Crop input to mouse start x
    // Mouse must be within plot bounds for this function to be called, so this is not checked again here
    Birch::Span<uint> xCrop(pixScaleX * (mousePos.X - plotLims.X.Min), 0);

    //printf("MouseZ: %lf\n", mousePos3.Z);

    // Crop input y to plot limits
    Birch::Span<uint> yCrop(0);

    if (dataYLimits.End > plotLims.Y.Max)
        yCrop.End = pixScaleY * (dataYLimits.End - plotLims.Y.Max);

    if (dataYLimits.Start < plotLims.Y.Min)
        yCrop.Start = pixScaleY * (plotLims.Y.Min - dataYLimits.Start);

    Birch::Point3<double> output(mousePos.X, mousePos.Y, 0);
    double bestValue = 0;

    //printf("MousePos: %lf, %lf, %lf\n", mousePos.X, mousePos.Y, mousePos3.Z);

    for (uint y = xCrop.Start; y < yRes - xCrop.End; y++) {
        for (uint x = yCrop.Start; x < xRes - yCrop.End; x++) {
            uint idx = y * xRes + x;
            
            auto dataPos = Birch::Point3<double>(dataXLimits.Start + y / pixScaleX, dataYLimits.Start + x / pixScaleY, log(data[idx]) * pwrScale);

            //printf("DataPos: %lf, %lf, %lf\n", dataPos.X, dataPos.Y, dataPos.Z);

            auto dist  = dataPos.DistanceSq(mousePos3);
            auto value = (data[idx]) / (dist + 1);

            if (value > bestValue) {
                bestValue = value;
                output    = dataPos;
            }
        }
    }

    //printf("Output: %lf, %lf, %lf\n", output.X, output.Y, output.Z);
    //printf("Dist: %lf\n", bestValue);

    // Round output to be centered on the pixel
    output.X = round(output.X * pixScaleX) / pixScaleX;// + dataXLimits.Length() / xRes / 2;
    output.Y = round(output.Y * pixScaleY) / pixScaleY;// + dataYLimits.Length() / yRes / 2;


    return Birch::Point<double>(output.X, output.Y);
}
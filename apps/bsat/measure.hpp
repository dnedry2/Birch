#ifndef __MEASURE_HPP__
#define __MEASURE_HPP__

#include "include/birch.h"
#include "defines.h"

#include "implot.h"

#include <string>

//#include "plot.hpp"

class PlotField;

enum class MeasureDisplay {
    None,
    X,
    Y,
    XY,
    DX,
    DY,
    DXY
};

// TODO: Rename to PlotMeasurement / make into abstract measurement class
class Measurement {
public:
    PlotField*            Field();          // Field that measurement is attached to
    Birch::Point<double>* Position();       // Position of measurement
    Birch::Span<double>*  Bounds();         // X Zoom where measurement was taken
    MeasureDisplay*       DisplayMode();    // Annotation display mode
    bool*                 Screen();         // Is measurement locked to a value
    llong*                DataIdx();
    char*                 XUnit();          // X axis unit of measurement
    char*                 YUnit();          // Y axis unit of measurement
    double*               XOffset();        // X offset for annotation display
    double*               YOffset();        // Y offset for annotation display
    std::string*          AltText();        // Alternative text to display for annotation

    Measurement(const Birch::Point<double>& position, const Birch::Span<double>& bounds, PlotField* field, bool screen, llong dataIdx, MeasureDisplay displayMode = MeasureDisplay::None, char* xUnit = nullptr, char* yUnit = nullptr);
    ~Measurement();

    Measurement(const Measurement&) = delete;
    Measurement& operator=(const Measurement&) = delete;

private:
    bool screen;
    Birch::Point<double> position;
    Birch::Span<double>  bounds;
    MeasureDisplay       displayMode;
    llong                dataIdx;
    char*                xUnit   = nullptr; // Should just point to some list of measurement units created elsewhere. Didn't want to have to copy it for every measurement
    char*                yUnit   = nullptr;
    double               xOffset = 0.0f;
    double               yOffset = 0.0f;
    std::string          altText = "";

    PlotField* field;
};

class HistMeasurement {
public:
    PlotField*           Field();          // Field that measurement is attached to
    Birch::Span<double>* Bounds();         // Start and end of selection
    float*               Curve();          // Curve of selection
    char*&               Stats();          // Stats of selection

    HistMeasurement(const Birch::Span<double>& bounds, PlotField* field);

private:
    Birch::Span<double> bounds;
    PlotField* field = nullptr;
    float*     curve = nullptr;
    char*      stats = nullptr;
};


// Searches for the closest point to the mouse in visual space
uint SmartSearch(const double* x, const double* y, const bool* mask, uint count, const Birch::Span<double>& yLimits, const ImPlotPoint& mousePos);
// Searches for the closest point to the mouse in visual space, within a box defined by the mouse position and tolerance
uint SmartSearchBox(const double* x, const double* y, const bool* mask, uint count, const ImPlotPoint& mousePos, uint pixTolerance);


Birch::Point<double> SmartSearchSpectra(const double* data, uint xRes, uint yRes, const Birch::Point<double>& mousePos, Birch::Span<double> dataXLimits, Birch::Span<double> dataYLimits, double maxValue);

#endif
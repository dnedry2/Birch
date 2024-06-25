/********************************
* Everything needed for colormaps
*
* Author: Levi Miller
* Date created: 1/27/2020
*********************************/

#ifndef _COLOR_
#define _COLOR_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include "imgui.h"
#include <GL/gl3w.h>
#include "math.h"
#include "gradient.hpp"

#include "decs.hpp"
#include "defines.h"

void InitColorWidgets(GLuint show, GLuint hide);

unsigned int* MakeRainbow(double f1, double f2, double f3, double p1, double p2, double p3, int size, int center = -1, int width = -1);
unsigned int* MakeColorGradient(unsigned int* colorStops, int stopCount, int size);

bool GradientPicker(const char* id, Gradient** gradient, ImVec2 size, std::vector<Gradient*>* gradients, Gradient** hovGradient, bool* hovered);

//represents a color layer
class Colorizer {
public:
    enum class Type { Point, Spectra };

    virtual Type Get_Type() = 0;
    virtual bool RenderWidget(int id, bool odd, bool range) = 0;
    virtual void Apply(uint* colormap, ulong size) = 0;

    bool Get_Enabled() { return enabled; }
    void Set_Enabled(bool enabled) { this->enabled = enabled; }

    // Returns the name of the reference field
    const char* FieldName() { return fieldName; }

    Colorizer(const char* fieldName) : fieldName(fieldName) { }
protected:
    const char* fieldName;
    const char* colorizerName;
    bool enabled = true;
};

class ColorizerSingle : public Colorizer {
public:
    Type Get_Type() override { return Type::Point; }
    bool RenderWidget(int id, bool odd, bool range) override;
    void Apply(uint* colormap, ulong size) override;

    ColorizerSingle(double* volatile* reference, char* referenceName, double value, double width, unsigned int color);

private:
    // Field to use for determining the color
    double* volatile* reference;
    // Color
    unsigned int color = 0xffffffff;
    // Center value to color on
    double value;
    // Distance +- value to color
    double width;
};

class ColorizerRange : public Colorizer {
public:
    Type Get_Type() override { return Type::Point; }
    bool RenderWidget(int id, bool odd, bool range) override;
    void Apply(uint* colormap, ulong size) override;

    ColorizerRange(double* volatile* reference, char* referenceName, double value, double width, Gradient* gradient, std::vector<Gradient*>* gradients);
private:
    double* volatile* reference;

    Gradient* gradient;
    std::vector<Gradient*>* gradients;

    unsigned int* renderedGradient;
    int renderSize = 256;

    // Center value to color on
    double value;
    // Distance +- value to color
    double width;

    bool lastHoverUpdate = false;
    Gradient* lastHoverGradient = nullptr;
};

class ColorizerSpectra : public Colorizer {
public:
    Type Get_Type() override { return Type::Spectra; }
    bool RenderWidget(int id, bool odd, bool range) override;
    void Apply(uint* colormap = nullptr, ulong size = 0) override;

    void Set_Color(Gradient* color, bool persist = true);
    Gradient* Get_Color();

    ColorizerSpectra(double* volatile* reference, unsigned* volatile* colormap, volatile unsigned long* volatile* colormapSize, const char* referenceName, volatile int* xRes, volatile int* yRes, volatile bool* ready, volatile char* version, Gradient* gradient, std::vector<Gradient*>* gradients, FilterStack* filters = nullptr);
    ~ColorizerSpectra();
private:
    double* volatile* reference = nullptr;
    unsigned* volatile* colormap = nullptr;
    volatile unsigned long* volatile* colormapSize = nullptr;
    volatile int* xRes = nullptr;
    volatile int* yRes = nullptr;
    volatile bool* ready = nullptr;
    volatile char* version = nullptr;

    Gradient* gradient = nullptr;
    bool transparent = false;
    bool autoscale = true;

    int as_smooth = 2;
    int as_blend = 0;

    float mag_min = 0;
    float mag_max = 100;

    float sig_mag_min = 0;
    float sig_mag_max = 0;
    
    float paretoDeg = 0;
    float paretoPreview[64];

    int scale = 0; //0 = Linear, 1 = NaturalLog

    volatile bool processing = false;

    //Gradient* gradient;
    std::vector<Gradient*>* gradients;
    //bool transparent = true;

    unsigned int* renderedGradient;
    int renderSize = 256;

    bool lastHoverUpdate = false;
    Gradient* lastHoverGradient = nullptr;

    int tempBufLen = 2048;
    double* tempBuf = nullptr;

    // Apply filters to the output
    bool filter = true;
    FilterStack* filters = nullptr;

    volatile bool cancelJob = false;
    std::thread* worker = nullptr;

    void expandBuffer(unsigned size);
    void apply();
};

class ColorStack {
public:
    std::vector<Colorizer*>* Get_Colors() { return &colors; }
    std::vector<ColorizerSpectra*>* Get_SpectraColors() { return &spectraColors; }
    void AddColor(Colorizer* color);
    Colorizer& operator[](int index) { return *colors[index]; }
    int Count() { return colors.size(); }
    unsigned int Get_BaseColor() { return baseColor; }
    void Set_BaseColor(unsigned int color) { baseColor = color; }
    void Set_SpectraColor(Gradient* color);

    void GenerateMap(uint* colormap, ulong size, bool* mask = nullptr);
    void GenerateSpectraMaps();
private:
    std::vector<Colorizer*> colors;
    std::vector<ColorizerSpectra*> spectraColors;
    unsigned int baseColor = 0xFFc9a279;
};

void TBDIQColor(PlotField* tbdpd);

#endif
/********************************
* Widgets - Contains various reusable UI widgets
*
* Author: Levi Miller
* Date created: 12/10/2020
*********************************/

#ifndef _WIDGETS_
#define _WIDGETS_

#include "decs.hpp"

#include <vector>
#include <algorithm>
#include <string>

#include <GL/gl3w.h>
#include "imgui.h"
#include "session.hpp"
#include "plot.hpp"
#include "gradient.hpp"
#include "include/birch.h"
#include "window.hpp"

#include "measure.hpp"

class IView;

// Enum from BFile { Signal, TBD }
enum class FileType;

// Sets
void InitWidgets(GLuint x, GLuint expand, GLuint hide, GLuint visible, GLuint hidden, GLuint search, GLuint input);

bool beginWidget(Widget *widget);
void endWidget();

// Base widget class
class Widget {
public:
    virtual void Render() = 0;
    virtual ~Widget() = 0;

    std::string Name();
    std::string SafeName();

    Widget(int id, GLuint icon, std::string name);

    IView* plotter = nullptr;
    bool Closable = false;
    bool Collapsed = false;
    bool Docked = false;
    int ID;
    const char* ToolID = nullptr; // Links widget to its tool counterpart. used to snap to widget when tool selected
    GLuint Icon = 0;

private:
    std::string name;
    std::string safeName;
};
// Plugin sidebar host
class PluginHost_Widget : public Widget {
public:
    PluginHost_Widget(Birch::PluginIQProcessor* plugin, int id);
    ~PluginHost_Widget();

    void Render() override;
private:
    Birch::PluginIQProcessor* plugin;
};
//sidebar file info panel
class Info_Widget : public Widget {
public:
    Info_Widget(GLuint icon, BFile* header, Session* session, int id);
    void Render() override;
private:
    BFile*   file    = nullptr;
    Session* session = nullptr;

    FileType type;
};

class Colormap_Widget : public Widget {
public:
    Colormap_Widget(int id, GLuint icon, Session* session, Gradient* defaultGradient, Gradient *defaultSpectraGradient, std::vector<Gradient*>* gradients);
    ImU32 Get_SelectedColor();
    Gradient* Get_SelectedGradient() { return selectedGradient; }
    ImU32 Get_BaseColor();
    Gradient* Get_SpectraColor();
    void Render() override;
    void AddColor(Colorizer* color);

    // Returns the version of the colormap. If the version is different from the last time it was checked, the colormap has changed.
    uint Version() { return version; }
    // Returns true if color tool should use solid color
    bool Solid() { return solid; }

    ~Colormap_Widget();

private:
    Session* session;
    std::vector<Gradient*>* gradients;
    PlotField* selectedField = nullptr;

    uint version = 0;

    double inputCenter = 0,
           inputWidth  = 0;

    bool swapFocus = false;
    bool solid     = false; // Add solid layer
    bool range     = false; // Range or center based input

    Gradient* selectedGradient;
    Gradient* spectraGradient;
    float selectedColor[3] = { 0.5f, 1.0f, 1.0f };
    float baseColor[3]     = { 0.5f, 0.75f, 1.0f };


    // Tooltips
    Tooltip fieldTip;
    Tooltip solidTip;
    Tooltip rangeTip;
};

class Filter_Widget : public Widget {
public:
    Filter_Widget(int id, GLuint icon, Session* session);
    bool Get_Link() { return linkIQ; }
    void Render();

    // Returns the version of the filter. If the version is different from the last time it was checked, the filter has changed.
    uint Version() { return version; }

    ~Filter_Widget();

private:
    Session* session;

    double inputCenter = 0,
           inputWidth  = 0;

    bool swapFocus = 0;

    bool linkIQ = false;

    unsigned version = 0;

    PlotField* selectedField = nullptr;
    int pass = 1;
    bool range = false;
    Tooltip rangeTip;
};

bool WindowWidget(WindowFunc** current, const std::vector<WindowFunc*>* windows);

bool BeginGroupBox(const char* name, const ImVec2& size);
void EndGroupBox();

bool RangeInput(double* center, double* width, bool dispRange, const ImVec2& size = ImVec2(0, 0));

#endif
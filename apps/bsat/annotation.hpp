#ifndef _ANNOT_
#define _ANNOT_

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include <GL/gl3w.h>

void InitAnnotations(GLuint note);

struct Annotation {
    Annotation(const char* text, double* position, double* bounds, int plot, int id, bool img = false);
    void Render(void* plot);

    char* Text;
    double Position[2];
    double Bounds[2];
    int PlotID;
    unsigned int Color = ImGui::ColorConvertFloat4ToU32(ImVec4(.2, .5, .2, .5));

    int ID;

private:
    char id[16];
    bool initial = true;

    GLuint img = 0;
};

#endif
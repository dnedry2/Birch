#ifndef _SVG_
#define _SVG_

#include <string>
#include <map>
#include "imgui.h"

// Loads an opengl texture from a svg string
unsigned int LoadSVG(const char* svg, int width, int height, float scale = 1);
// Loads a pixel array from a svg string
unsigned int* LoadSVGPixels(const char* svg, int width, int height, float scale = 1);

#endif
#include <stdlib.h>
#include <algorithm>

#include "nfd.h"
#include <GL/gl3w.h>

#include "lunasvg.h"

#include "svg.hpp"
#include "logger.hpp"

unsigned* LoadSVGPixels(const char* svg, int width, int height, float scale) {
    using namespace lunasvg;

    auto pix = Document::loadFromData(svg).get()->renderToBitmap(width, height);

    unsigned* pixels = new unsigned[width * height];
    std::copy_n((unsigned*)pix.data(), width * height, pixels);

    return pixels;
}

GLuint LoadSVG(const char *svg, int width, int height, float scale) {
    auto pix = LoadSVGPixels(svg, width, height, scale);

    GLuint tex;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix);

    delete[] pix;

    return tex;
}


#ifndef __FBO_HELPERS_H__
#define __FBO_HELPERS_H__

#include <GL/gl3w.h>

#include "imgui.h"
#include "imgui_internal.h"

ImDrawData makeDrawData(ImDrawList** dl, ImVec2 pos, ImVec2 size);
void renderDrawList(GLuint fbo, ImDrawList* dl, ImVec2 pos, ImVec2 size);
void makeFBO(ImVec2 size, GLuint* fbo, GLuint* tex);

#endif
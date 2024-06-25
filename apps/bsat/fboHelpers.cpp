#include "fboHelpers.hpp"
#include "imgui_impl_opengl3.h"

#include "stb_image_write.h"

/*
ImDrawData makeDrawData(ImDrawList** dl, ImVec2 pos, ImVec2 size) {
	ImDrawData draw_data = ImDrawData();

	draw_data.Valid = true;
	draw_data.CmdLists = dl;
	draw_data.CmdListsCount = 1;
	draw_data.TotalVtxCount = (*dl)->VtxBuffer.size();
	draw_data.TotalIdxCount = (*dl)->IdxBuffer.size();
	draw_data.DisplayPos = pos;
	draw_data.DisplaySize = size;
	draw_data.FramebufferScale = ImVec2(1, 1);

	return draw_data;
}

void renderDrawList(GLuint fbo, ImDrawList* dl, ImVec2 pos, ImVec2 size) {
    GLint last_texture, last_array_buffer, frameBuffer;
    glGetIntegerv(GL_TEXTURE_BINDING_2D,   &last_texture);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING,  &frameBuffer);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glViewport(0, 0, size.x, size.y);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	ImDrawData dd = makeDrawData(&dl, pos, size);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplOpenGL3_RenderDrawData(&dd);

	// write texture to file
	unsigned char* pixels = new unsigned char[size.x * size.y * 4];
	glReadPixels(0, 0, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	stbi_write_png("test.png", size.x, size.y, 4, pixels, size.x * 4);

	delete[] pixels;

    glBindTexture(GL_TEXTURE_2D, last_texture);
    glBindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
}
*/
void makeFBO(ImVec2 size, GLuint* fbo, GLuint* tex) {
    GLint last_texture, last_array_buffer, frameBuffer;
    glGetIntegerv(GL_TEXTURE_BINDING_2D,   &last_texture);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING,  &frameBuffer);

	GLuint color = 0;
	GLuint depth = 0;
	glGenFramebuffers(1, fbo);
	glGenTextures(1, &color);
	glGenRenderbuffers(1, &depth);

	glBindFramebuffer(GL_FRAMEBUFFER, *fbo);

	glBindTexture(GL_TEXTURE_2D, color);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);

	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size.x, size.y);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	*tex = color;

    glBindTexture(GL_TEXTURE_2D, last_texture);
    glBindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
}